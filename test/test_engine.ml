open Minigrad.Engine

let () =
  let x = make ~label:"x" 2.0 in
  let y = make ~label:"y" 3.0 in

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

  out.grad <- 1.0;
  backward out;

  Printf.printf "After backward:\n";
  show x;
  show y;
  show sum;
  show prod;
  show out
