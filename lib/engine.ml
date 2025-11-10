type value = {
  mutable data : float;
  mutable grad : float;
  op : string;
  prev : value list;
  mutable backward : unit -> unit;
  mutable label : string;
}
(** [value] : 用于构建计算图中的节点（神经网络中的标量节点）。
    - [value] 是自动微分系统（autograd engine）的基本节点，
    - [value] 支持任意复杂的数学表达式的自动求导（通过构建计算图并反向传播梯度）。

    @param data 节点储存的实际数值。
    @param grad 当前节点的梯度。
    @param op 生成该节点的操作符。
    @param prev 该节点的直接前驱节点集合。
    @param backward 会在每个有输入产生输出的小节点上执行链式法则，自动产生grad。
    @param label 可选标签。 *)

let make ?(label = "") (data : float) : value =
  { data; grad = 0.0; op = ""; prev = []; backward = (fun () -> ()); label }

let show (v : value) =
  Printf.printf "Value(label=%s, data=%.4f, grad=%.4f, op=%s)\n" v.label v.data
    v.grad v.op;
  flush stdout

(** [backward] : 自动拓扑反向传播*)
let rec backward v =
  v.backward ();
  List.iter backward v.prev

(** 算子重载 *)
let ( + ) a b =
  let out =
    {
      data = a.data +. b.data;
      grad = 0.0;
      backward = (fun () -> ());
      prev = [ a; b ];
      op = "+";
      label = "";
    }
  in
  out.backward <-
    (fun () ->
      a.grad <- a.grad +. out.grad;
      b.grad <- b.grad +. out.grad);
  out

let ( * ) a b =
  let out =
    {
      data = a.data *. b.data;
      grad = 0.0;
      backward = (fun () -> ());
      prev = [ a; b ];
      op = "*";
      label = "";
    }
  in
  out.backward <-
    (fun () ->
      a.grad <- a.grad +. (b.data *. out.grad);
      b.grad <- b.grad +. (a.data *. out.grad));
  out

let ( ~- ) a =
  let out =
    {
      data = -.a.data;
      grad = 0.0;
      op = "-";
      prev = [ a ];
      backward = (fun () -> ());
      label = "";
    }
  in
  out.backward <- (fun () -> a.grad <- a.grad -. out.grad);
  out

let ( - ) a b = a + ~-b

let tanh v =
  let y = Stdlib.tanh v.data in
  let out =
    {
      data = y;
      grad = 0.0;
      op = "tanh";
      prev = [ v ];
      backward = (fun () -> ());
      label = "";
    }
  in
  out.backward <-
    (fun () -> v.grad <- v.grad +. ((1.0 -. (y ** 2.0)) *. out.grad));
  out

let exp v =
  let y = Stdlib.exp v.data in
  let out =
    {
      data = y;
      grad = 0.0;
      op = "exp";
      prev = [ v ];
      backward = (fun () -> ());
      label = "";
    }
  in
  out.backward <- (fun () -> v.grad <- v.grad +. (y *. out.grad));
  out
