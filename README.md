# minigrad

![cat](./asset/cat.jpg)

**minigrad** is a simple OCaml refactoring of [microgard](https://github.com/karpathy/micrograd). It is a minimal automatic differentiation engine that implements reverse-mode automatic differentiation to handle dynamically constructed directed acyclic graphs (DAGs), and builds a small PyTorch-like neural network library on top of it.

It is **not recommended for educational purposes**, because the code is pretty messy ( ´･ω)

**minigrad**是对[microgard](https://github.com/karpathy/micrograd)使用ocaml的简单重构，是一个极小的仅可用的自动微分引擎，它实现了反向自动微分（反向模式自动微分）来处理动态构建的有向无环图（DAG），并在其上构建了一个类似PyTorch API的小型神经网络库。

**不建议用于教育，因为代码很烂 ( ´･ω) **

## Example 示例

Here is an example of basic values and operator operations:

下面是关于基本值和算子操作的示例：

```ocaml
(* Create values and assign labels *)
(* 创建值，对他加上标签 *)
let x = make ~label:"x" 2.0 in
let y = make ~label:"y" 3.0 in

(* Basic arithmetic with labels *)
(* 基本算数，标签赋值 *)
let sum = x + y in
sum.label <- "hi";
let prod = x * y in
let out = sum * prod in

Printf.printf "Before backward:\n";
show x;
show y;
show sum;
show prod;
show out;

(* Backpropagation *)
(* 向后传播 *)
out.grad <- 1.0;
backward out;

Printf.printf "After backward:\n";
show x;
show y;
show sum;
show prod;
show out
```

##  Neural Networks 神经网络

I implemented [a simple toy neural network in the neural network test](./test/test_nn.ml). I originally wanted to continue the demo from micrograd, but the most popular OCaml library, Owl, seems not to support macOS, so I cannot plot graphs. I haven’t continued from there, but maybe you can implement it—the materials are all here. ο(=•ω＜=)ρ⌒☆

我在[神经网络测试里](./test/test_nn.ml)实现了一个简单的菜数字网络，我本来想继续实现micrograd里的demo，但是ocaml最火爆的owl库似乎不支持macos，这让我无法画图，于是我再没有性质继续下去，也许你可以实现它，东西都摆在这里。ο(=•ω＜=)ρ⌒☆

## Running Tests 运行测试

Test files are located in `test/*`, including [test_engine](./test/test_engine.ml) for testing the engine, and [test_nn](./test/test_nn.ml) for testing the neural network. Running them requires only OCaml; no other dependencies are needed.

测试文件位于 `test/*`，分别是用于测试引擎的[test_engine](./test/test_engine.ml)和用于测试神经网络的[test_nn](./test/test_nn.ml)，运行他们不需要任何东西，只需要ocaml。

## License

BSD3