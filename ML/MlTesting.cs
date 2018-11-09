using System;
using System.IO;
using System.Linq;
using System.Text;

namespace ML
{
    public class MlTesting
    {
        public string RunClassifier()
        {
            DirectoryInfo imagesDir = new DirectoryInfo(Environment.CurrentDirectory + "/data");
            if (!imagesDir.Exists || !imagesDir.EnumerateFiles().Any())
            {
                throw new Exception("Directory was not found: " + imagesDir.FullName);
            }

            var modelScorer = new TFModelScorer(
                "data/tags.tsv",
                imagesDir.FullName,
                "model/tensorflow_inception_graph.pb",
                "model/imagenet_comp_graph_label_strings.txt");

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