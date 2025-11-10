(* 测试神经网络模块的完整示例 *)
open Minigrad.Engine (* 导入引擎模块，提供自动求导功能 *)
open Minigrad.Nn (* 导入神经网络模块，包含MLP等组件 *)

let () =
  (* 训练数据：4 个样本，每个样本 3 个特征 *)
  (* xs 是一个列表的列表，外层列表包含4个样本，内层列表每个有3个特征值 *)
  let xs =
    [
      [ 2.0; 3.0; -1.0 ];
      (* 样本1 *)
      [ 3.0; -1.0; 0.5 ];
      (* 样本2 *)
      [ 0.5; 1.0; 1.0 ];
      (* 样本3 *)
      [ 1.0; 1.0; -1.0 ];
      (* 样本4 *)
    ]
  in

  (* 对应标签/目标值：4个样本的真实输出值 *)
  (* ys 是一个浮点数列表，与xs中的样本一一对应 *)
  let ys = [ 1.0; -1.0; -1.0; 1.0 ] in

  (* 创建一个多层感知机(MLP)神经网络 *)
  (* MLP.make 函数用法：MLP.make 输入层神经元数 [隐藏层1神经元数; 隐藏层2神经元数; ...; 输出层神经元数] *)
  (* 这里创建了一个3-4-4-1的网络结构： *)
  (* - 输入层：3个神经元（对应3个特征） *)
  (* - 隐藏层1：4个神经元 *)
  (* - 隐藏层2：4个神经元 *)
  (* - 输出层：1个神经元（单输出） *)
  let n = MLP.make 3 [ 4; 4; 1 ] in

  (* 训练循环，迭代100次（100个epoch） *)
  for epoch = 0 to 99 do
    (* 前向传播：计算每个样本的预测值 *)
    (* List.map 函数用法：List.map (函数) 列表 *)
    (* 对xs中的每个样本应用函数，返回预测值的列表 *)
    let ypred =
      List.map
        (fun xrow ->
          (* 把普通float列表转换为可求导的value类型列表 *)
          (* List.map 再次使用，将每个特征值转换为value类型 *)
          let inputs = List.map make xrow in

          (* 将输入传入MLP网络进行前向传播计算 *)
          (* MLP.forward 函数用法：MLP.forward 网络 输入列表 -> 输出列表 *)
          match MLP.forward n inputs with
          | [ y ] -> y (* 由于输出层只有1个神经元，取列表的第一个元素 *)
          | _ -> failwith "MLP output dimension mismatch") (* 输出维度不匹配时抛出异常 *)
        xs
    in

    (* 计算损失：均方误差 (MSE) *)
    (* List.map2 函数用法：List.map2 (函数) 列表1 列表2 *)
    (* 对两个列表（真实值ys和预测值ypred）中的对应元素应用函数，返回新的列表 *)
    (* 这里用于计算每个样本的平方误差 *)
    let losses =
      List.map2
        (fun ygt yout ->
          let ytrue = make ygt in
          (* 将真实标签转换为value类型 *)
          let diff = yout - ytrue in
          (* 计算预测值与真实值的差值 *)
          diff * diff) (* 计算平方误差 *)
        ys ypred
    in

    (* 将所有样本的损失求和，得到总损失 *)
    (* List.fold_left 函数用法：List.fold_left (累加函数) 初始值 列表 *)
    (* 从左到右遍历列表，将每个元素累加到初始值上 *)
    (* 这里将所有样本的损失相加，得到总损失 *)
    let loss = List.fold_left (fun acc l -> acc + l) (make 0.0) losses in

    (* 清零所有参数的梯度，准备下一步反向传播 *)
    (* List.iter 函数用法：List.iter (函数) 列表 *)
    (* 对列表中的每个元素应用函数（无返回值） *)
    (* 这里将所有参数的梯度重置为0 *)
    List.iter (fun p -> p.grad <- 0.0) (MLP.parameters n);

    (* 设置总损失的梯度为1，触发反向传播链 *)
    (* 这是反向传播的起点，梯度从loss开始向后传播 *)
    loss.grad <- 1.0;

    (* 执行反向传播，自动计算每个参数的梯度 *)
    (* backward 函数会沿着计算图反向传播梯度 *)
    backward loss;

    (* 参数更新：使用梯度下降法 *)
    (* 再次使用List.iter遍历所有参数，根据梯度更新参数值 *)
    (* 学习率为0.01，参数更新公式：新参数 = 旧参数 - 学习率 * 梯度 *)
    List.iter (fun p -> p.data <- p.data -. (0.01 *. p.grad)) (MLP.parameters n);

    (* 每10个epoch打印训练信息 *)
    if epoch mod 10 = 0 then begin
      (* 打印当前epoch和损失值 *)
      Printf.printf "Epoch %d, Loss: %.4f\n" epoch loss.data
    end
  done
