using System;


namespace ML.Console
{
    class Program
    {
        static void Main(string[] args)
        {
            var ml = new ML.MlTesting();
            var result = ml.RunClassifier();
            System.Console.Write(result);
            System.Console.ReadKey();
        }
    }
}
