var json = require('./report.json');
console.log("[");
json.forEach(function(item) {
  console.log("" + item.run + "," + item.epoch + "," + item.accuracy +";");
});
console.log("]");
