import torch
import inspect
import types
from functools import wraps
from .graph import Graph


class OpTensor(object):
    def __init__(self, inp, name=''):
        self.name = name
        self.type = type(inp)
        self.istensor = isinstance(inp, torch.Tensor)
        if self.istensor:
            self.shape = inp.shape
            self.dtype = inp.dtype
            self.id = inp.data_ptr()

    def __repr__(self):
        res = "name: {}, type: {}".format(self.name, self.type)
        if self.istensor:
            res1 = ", Tensor: shape {}, dtype {}, id {}".format(self.shape, self.dtype, self.id)
            res += res1
        return res


class TorchOp(object):
    def __init__(self, args=[], output=None):
        self.inputs = [OpTensor(arg) for arg in args]
        self.output = OpTensor(output) if output is not None else None

    @property
    def op_name(self):
        raise NotImplementedError

    def __repr__(self):
        res = "({} - {}, \ninputs:\n".format(self.__class__.__name__, self.op_name)
        for i, inp in enumerate(self.inputs):
            res += '\tinput{} - '.format(i) + inp.__repr__()
        res += "\n output: "
        if self.output is not None:
            res += self.output.__repr__()
        res += "\n)"
        return res


class FuncOp(TorchOp):
    def __init__(self, func, args, output):
        super(FuncOp, self).__init__(args, output)
        self._op_name = func.__name__

    @property
    def op_name(self):
        return self._op_name


class ModuleOp(TorchOp):
    def __init__(self, module, args, output):
        super(ModuleOp, self).__init__(args, output)
        self.module = module
        self.ops = []

    @property
    def op_name(self):
        return self.module.__class__.__name__


class Nop(TorchOp):
    def __init__(self, name):
        super(Nop, self).__init__()
        self._op_name = name
        self.name = name

    @property
    def op_name(self):
        return self._op_name


class TorchTracer(object):
    def __init__(self):
        self._trace = []
        self._main_trace = None

    def __enter__(self):
        self.trace_ = []
        self.torch = types.SimpleNamespace()
        self.Tensor = types.SimpleNamespace()
        setattr(self.torch, 'funcs', [])
        setattr(self.Tensor, 'funcs', [])

        # Wrap torch.xxx functions
        for name in dir(torch._C._VariableFunctions):
            if name.startswith('__') or name.startswith('_'):
                continue
            if hasattr(torch, name):
                #             print(func.__name__)
                func = getattr(torch, name)
                self.torch.funcs.append(name)
                setattr(self.torch, name, func)
                setattr(torch, name, self.wrap_func(func))

        # Wrap torch.Tensor methods
        tensor_methods = self._get_tensor_methods()
        for name, func in tensor_methods:
            if hasattr(torch.Tensor, name):
                self.Tensor.funcs.append(name)
                setattr(self.Tensor, name, func)
                setattr(torch.Tensor, name, self.wrap_func(func))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name in self.torch.funcs:
            setattr(torch, name, getattr(self.torch, name))
        for name in self.Tensor.funcs:
            setattr(torch.Tensor, name, getattr(self.Tensor, name))

    def _get_tensor_methods(self):
        exclude_methods = ['__format__',
                           '__dir__',
                           '__sizeof__',
                           '_is_view',
                           '_make_subclass',
                           '_values',
                           'data_ptr',
                           'type',
                           'type_as']

        wrapper_descriptor = type(torch.Tensor.__getattribute__)
        all_methods = inspect.getmembers(torch.Tensor, predicate=inspect.isroutine)
        tensor_methods = [f for f in all_methods if type(f[1]) != wrapper_descriptor and f[0] not in exclude_methods]
        return tensor_methods

    def wrap_func(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            #             print(func.__qualname__)
            result = func(*args, **kwargs)
            op = FuncOp(func, list(args), result)
            self.trace.append(op)
            return result

        return wrapper

    @property
    def trace(self):
        return self._trace

    def redirect_trace(self, op_list):
        self._main_trace = self._trace
        self._trace = op_list

    def restore_tracing(self):
        if self._main_trace is not None:
            self._trace = self._main_trace
            self._main_trace = None

    def trace_model(self, model, input):
        module_ops = {}

        def pre_hook(module, input):
            module_ops[module] = []
            self.redirect_trace(module_ops[module])

        def hook(module, input, output):
            self.restore_tracing()
            mop = ModuleOp(module, input, output)
            mop.ops = module_ops[module]
            self.trace.append(mop)

        leafs = [m for m in model.modules() if len([m for m in m.children()]) == 0]
        #     print(leafs)

        handles = []
        for m in leafs:
            handles.append(m.register_forward_pre_hook(pre_hook))
            handles.append(m.register_forward_hook(hook))

        model(input)

        for h in handles:
            h.remove()

    def to_graph(self):
        return TorchTracer.trace_to_graph(self.trace)

    def node_to_graph(self, node):
        if isinstance(node, ModuleOp):
            return TorchTracer.trace_to_graph(node.ops)
        elif isinstance(node, FuncOp):
            return TorchTracer.trace_to_graph([node])
        elif isinstance(node, Nop):
            g = Graph()
            g.add_node(node)
            return g
        else:
            raise ValueError("node is invalid")

    @staticmethod
    def trace_to_graph(trace):
        # Create unique names for ops
        op_counter = {}
        for op in trace:
            if op.op_name in op_counter:
                op_counter[op.op_name] += 1
            else:
                op_counter[op.op_name] = 0

            op.name = "{}{}".format(op.op_name, op_counter[op.op_name])

        tensor_counter = const_counter = scalar_counter = 0
        # Find all connections
        conn = {}
        for op in trace:
            for inp in op.inputs:
                if inp.istensor:
                    if inp.id not in conn:
                        conn[inp.id] = type('', (object,), {"consumers": [], "producers": []})()
                        conn[inp.id].type = inp.type

                    conn[inp.id].consumers.append(op)
                else:
                    conn[id(inp)] = type('', (object,),
                                         {"consumers": [op], "producers": [Nop('const{}'.format(const_counter))]})()
                    const_counter += 1

            if op.output.istensor:
                if op.output.id not in conn:
                    conn[op.output.id] = type('', (object,), {"consumers": [], "producers": []})()
                    conn[op.output.id].type = op.output.type

                conn[op.output.id].producers.append(op)
            else:
                conn[id(op.output)] = type('', (object,),
                                           {"consumers": [Nop('scalar{}'.format(scalar_counter))], "producers": [op]})()
                scalar_counter += 1

        # create input/output nodes for not connected tensors
        for e in conn:
            if len(conn[e].consumers) == 0:
                conn[e].consumers.append(Nop('{}{}'.format(conn[e].type.__name__, tensor_counter)))
                tensor_counter += 1
            if len(conn[e].producers) == 0:
                conn[e].producers.append(Nop('{}{}'.format(str(conn[e].type.__name__), tensor_counter)))
                tensor_counter += 1

        # create graph from connections
        g = Graph()

        # add ops as nodes
        for tid in conn:
            for c in conn[tid].consumers:
                g.add_node(c)
            for p in conn[tid].producers:
                g.add_node(p)

        # add tensor connections as adges to graph
        for e in conn:
            for p in conn[e].producers:
                for c in conn[e].consumers:
                    g.add_edge(p, c)

        return g
