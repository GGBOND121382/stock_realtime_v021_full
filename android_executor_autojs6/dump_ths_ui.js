"use strict";
auto.waitFor();
console.show();
app.launchPackage("com.hexin.plat.android");
sleep(500);
var nodes = [
  "com.hexin.plat.android:id/btn",
  "com.hexin.plat.android:id/auto_stockcode",
  "com.hexin.plat.android:id/dialogplus_view_container",
  "com.hexin.plat.android:id/stockcode_tv",
  "com.hexin.plat.android:id/stockvolume",
  "com.hexin.plat.android:id/stockprice",
  "com.hexin.plat.android:id/btn_transaction",
  "com.hexin.plat.android:id/ok_btn",
  "com.hexin.plat.android:id/cancel_btn",
  "com.hexin.plat.android:id/prompt_content"
];
nodes.forEach(function (rid) {
  var count = id(rid).find().size();
  console.log(rid + " count=" + count);
});
console.log("当前包: " + currentPackage());
console.log("当前Activity: " + currentActivity());
toast("已输出同花顺控件探测结果");
