using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Transforms;

namespace ML
{
    public class TFModelScorer
    {
        private readonly string dataLocation;
        private readonly string imagesFolder;
        private readonly string modelLocation;
        private readonly string labelsLocation;
        private readonly MLContext mlContext;

        public TFModelScorer(string dataLocation, string imagesFolder, string modelLocation, string labelsLocation)
        {
            this.dataLocation = dataLocation;
            this.imagesFolder = imagesFolder;
            this.modelLocation = modelLocation;
            this.labelsLocation = labelsLocation;
            mlContext = new MLContext();
        }

        public struct ImageNetSettings
        {
            public const int imageHeight = 224;
            public const int imageWidth = 224;
            public const float mean = 117;
            public const bool channelsLast = true;
        }

        public struct InceptionSettings
        {
            // for checking tensor names, you can use tools like Netron,
            // which is installed by Visual Studio AI Tools

            // input tensor name
            public const string inputTensorName = "input";

            // output tensor name
            public const string outputTensorName = "softmax2";
        }

        public IEnumerable<ImageNetDataProbability> Score()
        {
            var model = LoadModel(dataLocation, imagesFolder, modelLocation);

            return PredictDataUsingModel(dataLocation, imagesFolder, labelsLocation, model).ToArray();
        }

        private PredictionFunction<ImageNetData, ImageNetPrediction> LoadModel(string dataLocation, string imagesFolder, string modelLocation)
        {
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Training file: {dataLocation}");
            Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth},{ImageNetSettings.imageHeight}), image mean: {ImageNetSettings.mean}");

            var loader = new TextLoader(mlContext,
                new TextLoader.Arguments
                {
                    Column = new[] {
                        new TextLoader.Column("ImagePath", DataKind.Text, 0),
                    }
                });

            var data = loader.Read(new MultiFileSource(dataLocation));

            var pipeline = ImageEstimatorsCatalog.LoadImages(catalog: mlContext.Transforms, imageFolder: imagesFolder, columns: ("ImagePath", "ImageReal"))
                .Append(ImageEstimatorsCatalog.Resize(mlContext.Transforms, "ImageReal", "ImageReal", ImageNetSettings.imageHeight, ImageNetSettings.imageWidth))
                .Append(ImageEstimatorsCatalog.ExtractPixels(mlContext.Transforms, new[] { new ImagePixelExtractorTransform.ColumnInfo("ImageReal", InceptionSettings.inputTensorName, interleave: ImageNetSettings.channelsLast, offset: ImageNetSettings.mean) }))
                .Append(new TensorFlowEstimator(mlContext, modelLocation, new[] { InceptionSettings.inputTensorName }, new[] { InceptionSettings.outputTensorName }));

            var modeld = pipeline.Fit(data);

            var predictionFunction = modeld.MakePredictionFunction<ImageNetData, ImageNetPrediction>(mlContext);
            return predictionFunction;
        }

        protected IEnumerable<ImageNetDataProbability> PredictDataUsingModel(string testLocation, string imagesFolder, string labelsLocation, PredictionFunction<ImageNetData, ImageNetPrediction> model)
        {
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Training file: {testLocation}");
            Console.WriteLine($"Labels file: {labelsLocation}");

            var labels = File.ReadAllLines(labelsLocation);

            var testData = ImageNetData.ReadFromTsv(testLocation, imagesFolder);

            foreach (var sample in testData)
            {
                var probs = model.Predict(sample).PredictedLabels;
                var imageData = new ImageNetDataProbability()
                {
                    ImagePath = sample.ImagePath,
                    Label = sample.Label
                };
                (imageData.PredictedLabel, imageData.Probability) = GetBestLabel(labels, probs);

                yield return imageData;
            }
        }


    
        public static (string, float) GetBestLabel(string[] labels, float[] probs)
        {
            var max = probs.Max();
            var index = probs.AsSpan().IndexOf(max);
                return (labels[index], max);
        }
    }

    public class ImageNetDataProbability : ImageNetData
    {
        public string PredictedLabel;
        public float Probability { get; set; }
    }

    public class ImageNetPrediction
    {
        [ColumnName(TFModelScorer.InceptionSettings.outputTensorName)]
        public float[] PredictedLabels;
    }
}