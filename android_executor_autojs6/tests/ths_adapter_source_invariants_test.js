"use strict";

var assert = require("assert");
var fs = require("fs");
var path = require("path");

var sourcePath = path.join(__dirname, "..", "lib", "ths_adapter.js");
var src = fs.readFileSync(sourcePath, "utf8");

function bodyOf(functionName) {
  var marker = "function " + functionName + "(";
  var start = src.indexOf(marker);
  assert.ok(start >= 0, "missing function: " + functionName);
  var brace = src.indexOf("{", start);
  assert.ok(brace >= 0, "missing body: " + functionName);
  var depth = 0;
  for (var i = brace; i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") {
      depth--;
      if (depth === 0) return src.slice(brace + 1, i);
    }
  }
  throw new Error("unterminated function body: " + functionName);
}

function count(haystack, needle) {
  var n = 0;
  var pos = 0;
  while ((pos = haystack.indexOf(needle, pos)) >= 0) {
    n++;
    pos += needle.length;
  }
  return n;
}

assert.strictEqual(src.indexOf("FILL_ATTEMPTS"), -1, "whole-order fill retry must stay removed");
assert.strictEqual(src.indexOf("fillOrderFieldsOnce"), -1, "no duplicate whole-order fill layer");
assert.strictEqual(/(^|[^.\w])setText\s*\(/m.test(src), false, "global setText() must never be used");

var writeBody = bodyOf("setNodeTextOnce");
assert.strictEqual(count(writeBody, ".setText("), 1, "write helper must issue exactly one node-scoped setText");

var codeBody = bodyOf("fillCode");
assert.strictEqual(count(codeBody, "setNodeTextOnce(searchEdit, expected"), 1, "stock code must be written exactly once");
assert.strictEqual(count(codeBody, "tapCenter(suggestion)"), 1, "stock suggestion must have only one tap site");
assert.strictEqual(codeBody.indexOf("safeNodeText(searchEdit)"), -1, "stale search UiObject must not be read back");

var fillBody = bodyOf("fillOrderFields");
assert.strictEqual(count(fillBody, "setNodeTextOnce(volumeEdit, qtyText"), 1, "quantity must be written exactly once");
assert.strictEqual(count(fillBody, "setNodeTextOnce(priceEdit, priceText"), 1, "price must be written exactly once");
assert.strictEqual(fillBody.indexOf("recoverToTradingPage"), -1, "failed order must not be retried inside fillOrderFields");

console.log("ths_adapter source invariants: OK");
