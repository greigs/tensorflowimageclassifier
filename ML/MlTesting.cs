using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

namespace ML
{
    public class MlTesting
    {
        public string RunClassifier()
        {
            var baseDirectory = new FileInfo(Assembly.GetExecutingAssembly().Location).Directory;
            var imagesDir = new DirectoryInfo(baseDirectory + "/data");
            var modelDir = new DirectoryInfo(baseDirectory + "/model");
            if (!imagesDir.Exists || !imagesDir.EnumerateFiles().Any())
            {
                throw new Exception("Directory was not found: " + imagesDir.FullName);
            }

            if (!modelDir.Exists || !modelDir.EnumerateFiles().Any())
            {
                throw new Exception("Directory was not found: " + modelDir.FullName);
            }


            var modelScorer = new TFModelScorer(
                imagesDir + "/tags.tsv",
                imagesDir.FullName,
                modelDir + "/tensorflow_inception_graph.pb",
                modelDir + "/imagenet_comp_graph_label_strings.txt");

            var results = modelScorer.Score();
            var sb = new StringBuilder();
            foreach (var imageNetData in results)
            {
                sb.AppendLine("Input: " + imageNetData.Label);
                sb.AppendLine("Prediction: " + imageNetData.PredictedLabel);
                sb.AppendLine("Probability: " + imageNetData.Probability.ToString());
                sb.AppendLine();
            }

            return sb.ToString();
        }
    }
}