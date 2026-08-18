"use strict";

var assert = require("assert");
var contract = require("../lib/order_contract.js");

var order = { side: "buy", code: "600521", qty: 300, submit_price: 16.06 };
var parsed = contract.parseConfirmationTexts([
  "委托买入确认",
  "账户", "A123456",
  "名称", "测试证券",
  "代码", "600521",
  "数量", "300",
  "价格", "16.060"
]);
assert.strictEqual(parsed.code, "600521");
assert.strictEqual(parsed.qty, "300");
assert.strictEqual(parsed.price, "16.060");
assert.strictEqual(contract.validateConfirmation(order, parsed).ok, true);

var inline = contract.parseConfirmationTexts([
  "委托买入确认",
  "证券代码：600521.SH",
  "委托数量：300",
  "委托价格：16.06"
]);
assert.strictEqual(contract.validateConfirmation(order, inline).ok, true);

var sellOrder = { side: "sell", code: "600521", qty: 13, submit_price: 16.06 };
var sellParsed = contract.parseConfirmationTexts([
  "委托卖出确认", "代码", "600521", "数量", "13", "价格", "16.06"
]);
assert.strictEqual(contract.validateConfirmation(sellOrder, sellParsed).ok, true);
assert.strictEqual(contract.validateConfirmation(order, sellParsed).ok, false);

var wrongCode = contract.parseConfirmationTexts([
  "委托买入确认", "代码", "600522", "数量", "300", "价格", "16.06"
]);
assert.strictEqual(contract.validateConfirmation(order, wrongCode).ok, false);

var wrongQty = contract.parseConfirmationTexts([
  "委托买入确认", "代码", "600521", "数量", "16", "价格", "16.06"
]);
assert.strictEqual(contract.validateConfirmation(order, wrongQty).ok, false);

var wrongPrice = contract.parseConfirmationTexts([
  "委托买入确认", "代码", "600521", "数量", "300", "价格", "300"
]);
assert.strictEqual(contract.validateConfirmation(order, wrongPrice).ok, false);

var missingField = contract.parseConfirmationTexts([
  "委托买入确认", "代码", "600521", "数量", "300"
]);
assert.strictEqual(contract.validateConfirmation(order, missingField).ok, false);

assert.strictEqual(
  contract.validateConfirmation({ side: "hold", code: "600521", qty: 300, submit_price: 16.06 }, parsed).ok,
  false
);

assert.strictEqual(contract.classifyResultMessage("委托已提交，合同号为：123456").outcome, "submitted");
assert.strictEqual(contract.classifyResultMessage("股票余额不足 ,不允许卖空").outcome, "rejected");
assert.strictEqual(contract.classifyResultMessage("委托价格不合法，请重新输入").outcome, "rejected");
assert.strictEqual(contract.classifyResultMessage("网络繁忙，请稍后再试").outcome, "unknown");
assert.strictEqual(contract.classifyResultMessage("委托未提交，请稍后查询").outcome, "unknown");
assert.strictEqual(contract.classifyResultMessage("委托已提交失败，请重新确认").outcome, "unknown");
assert.strictEqual(contract.classifyResultMessage("提交委托失败，请查询委托记录").outcome, "unknown");

console.log("order_contract tests: OK");
