import Foundation
import CreateML

// Import training data
// let trainSet = URL(fileURLWithPath: "/Users/danielsogl/Projekte/DHBW/Sentiment Analysis/data/classification/train")
let data =  try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/danielsogl/Desktop/Office_Products_5.json"))
let (trainingData, testingData) = data.randomSplit(by: 0.8, seed: 5)

trainingData.size
testingData.size

// Train model
let model = try MLTextClassifier(trainingData: trainingData,
                                 textColumn: "reviewText",
                                 labelColumn: "overall")

// Show training metrics
model.trainingMetrics

// Training accuracy as a percentage
let trainingAccuracy = (1.0 - model.trainingMetrics.classificationError) * 100

// Validation accuracy as a percentage
let validationAccuracy = (1.0 - model.validationMetrics.classificationError) * 100

let evaluationMetrics = model.evaluation(on: testingData)

// Evaluation accuracy as a percentage
let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100

let metadata = MLModelMetadata(author: "Daniel Sogl",
                               shortDescription: "A model trained to classify office product review sentiment",
                               version: "1.0")

try model.write(to: URL(fileURLWithPath: "/Users/danielsogl/Projekte/DHBW/Sentiment Analysis/models/SentimentClassifier.mlmodel"),
                              metadata: metadata)
