using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime.Api;

namespace ML
{
    public class ImageNetData
    {
        [Column("0")]
        public string ImagePath;

        [Column("1")]
        public string Label;

        public static IEnumerable<ImageNetData> ReadFromTsv(string file, string folder)
        {
            return File.ReadAllLines(file)
                .Select(x => x.Split('\t'))
                .Select(x => new ImageNetData()
                {
                    ImagePath = Path.Combine(folder, x[0]),
                    Label = x[1],
                });
        }
    }
}