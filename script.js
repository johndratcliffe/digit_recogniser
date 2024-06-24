let results
const csvData = Papa.parse('/train.csv', {
  download: true,
  header: true,
  complete: function (data) {
    results = data.data
    console.log(results)
  }
})
