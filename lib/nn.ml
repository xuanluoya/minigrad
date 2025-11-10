open Engine

module Neuron = struct
  type t = { mutable w : value list; mutable b : value }

  let make nin =
    Random.self_init ();
    (* 防止梯度爆炸，输入的nin越大，每个输入对输出的影响越小，所以权重要更小。 *)
    let scale = 1.0 /. sqrt (float_of_int nin) in
    let w =
      List.init nin (fun _ ->
          let r = (Random.float 2.0 -. 1.0) *. scale in
          make r)
    in
    let b = make ((Random.float 2.0 -. 1.0) *. 0.1) in
    { w; b }

  let forward (v : t) (x : value list) : value =
    (* z = Σ (w_i * x_i) + b *)
    let sum =
      (* v.w 和 x 按位相乘 *)
      (* 把累加器 acc 与 wi * xi 相加得到新的累加器。这里 wi 来自第一个列表（v.w 权重），xi 来自第二个列表（x 输入） *)
      List.fold_left2 (fun acc wi xi -> acc + (wi * xi)) (Engine.make 0.0) v.w x
    in
    let act = sum + v.b in
    (* 激活非线性 *)
    tanh act

  (** 获取神经元参数，拼接两个列表*)
  let parameters v = v.w @ [ v.b ]
end

module Layer = struct
  type t = { neurons : Neuron.t list }

  let make nin nout =
    let neurons = List.init nout (fun _ -> Neuron.make nin) in
    { neurons }

  let forward (l : t) (x : value list) : value list =
    List.map (fun n -> Neuron.forward n x) l.neurons

  (** 提取每个layer里每个参数，扁平化成一个列表。 *)
  let parameters layer = List.flatten (List.map Neuron.parameters layer.neurons)
end

module MLP = struct
  type t = { layers : Layer.t list }

  (* 构造函数: nin = 输入大小, nouts = 每层神经元数量的列表 *)
  let make nin nouts =
    (* 把nin拼nouts前面 *)
    let sz = nin :: nouts in
    (* 构建每一层 *)
    let layers =
      (* (List.length nouts) : 一共几层 *)
      (* i 是当前层的索引 *)
      List.init (List.length nouts) (fun i ->
          (* List.nth -> 按索引访问列表中的元素 *)
          let nin = List.nth sz i in
          let nout = List.nth sz Stdlib.(i + 1) in
          Layer.make nin nout)
    in
    { layers }

  (* 前向传播 *)
  let forward (mlp : t) (x : value list) : value list =
    (* fold_left 初始值 x 遍历layer全部进行向前传播 输入是上一层的输出（累积值 input）返回当前层的输出（下一层输入）*)
    List.fold_left (fun input layer -> Layer.forward layer input) x mlp.layers

  (* 获取所有参数 *)
  let parameters mlp = List.flatten (List.map Layer.parameters mlp.layers)
end
