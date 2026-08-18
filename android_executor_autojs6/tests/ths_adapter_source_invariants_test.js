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

// No whole-order fill retry is allowed. A failure must surface its exact stage.
assert.strictEqual(src.indexOf("FILL_ATTEMPTS"), -1, "whole-order fill retry must stay removed");
var fillBody = bodyOf("fillOrderFields");
assert.strictEqual(fillBody.indexOf("recoverToTradingPage"), -1, "fillOrderFields must not recover-and-retry");
assert.strictEqual(fillBody.indexOf("for ("), -1, "fillOrderFields must not loop");
assert.strictEqual(fillBody.indexOf("while ("), -1, "fillOrderFields must not loop");

// The only write helper is node-scoped and performs exactly one UiObject#setText call.
var writeBody = bodyOf("setNodeTextOnce");
assert.strictEqual(count(writeBody, ".setText("), 1, "setNodeTextOnce must issue exactly one node-scoped setText");
assert.strictEqual(/(^|[^.\w])setText\s*\(/m.test(src), false, "global setText() must never be used");

// One code write, one quantity write, one price write per fill attempt.
var codeBody = bodyOf("fillCode");
assert.strictEqual(count(codeBody, "setNodeTextOnce(searchEdit, expected"), 1, "stock code must be written exactly once");
var fillOnceBody = bodyOf("fillOrderFieldsOnce");
assert.strictEqual(count(fillOnceBody, "setNodeTextOnce(volumeEdit, qtyText"), 1, "quantity must be written exactly once");
assert.strictEqual(count(fillOnceBody, "setNodeTextOnce(priceEdit, priceText"), 1, "price must be written exactly once");

console.log("ths_adapter source invariants: OK");
