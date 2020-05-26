using System;
using System.Linq;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            //~~~~~~~~~~~~~~~~~~~~ irisy ~~~~~~~~~~~~~~~~~~~~
            DataSet Data = new DataSet("...\\...\\iris.data");
            NeuralNetwork network = new NeuralNetwork(new int[] { 4, 3, 2, 3 });
            network.ShowNetwork();
            int gen = 2500;
            for (int i = 0; i < gen; i++)
            {
                for (int j = 0; j < Data.LearningData.Length; j++)
                {
                    network.Learning(Data.LearningData[j], Data.LearningOutput[j]);
                }
            }

            network.ShowNetwork();

            int pass1 = 0;
            for (int i = 0; i < Data.TestData.Length; i++)
            {
                if (network.FeedForward(Data.TestData[i], Data.TestOutput[i]))
                    pass1++;
            }
            Console.WriteLine($"All = {Data.TestData.Length}, Correct = {pass1} Procentage = {100 * pass1 / Data.TestData.Length}%");


            //~~~~~~~~~~~~~~~~~~~~ Adult dataset ~~~~~~~~~~~~~~~~~~~~
            //DataSet2 dataset2 = new DataSet2();
            //dataset2.NormalizeData();
            //NeuralNetwork n = new NeuralNetwork(new int[] { 14, 10, 10, 1 });
            //n.ShowNetwork();
            //int gen = 10;
            //for (int j = 0; j < gen; j++)
            //{
            //    for (int i = 0; i < dataset2.LearningTest.Length; i++)
            //    {
            //        n.Learning(dataset2.LearningTest[i].Take(dataset2.LearningTest[i].Length - 1).ToArray(), dataset2.LearningTest[i].Skip(dataset2.LearningTest[i].Length - 1).ToArray());
            //    }
            //}
            //n.ShowNetwork();

            //int pass = 0;
            //for (int i = 0; i < dataset2.TestSet.Length; i++)
            //{
            //    if (n.FeedForward(dataset2.TestSet[i].Take(dataset2.TestSet[i].Length - 1).ToArray(), dataset2.TestSet[i].Skip(dataset2.TestSet[i].Length - 1).ToArray()))
            //        pass++;
            //}
            //Console.WriteLine($"All = {dataset2.TestSet.Length}, Correct = {pass} Procentage = {100 * pass / dataset2.TestSet.Length}%");


            //~~~~~~~~~~~~~~~~~~~~ XOR ~~~~~~~~~~~~~~~~~~~~
            //double[][] input = new double[][] { new double[] { 1, 1 }, new double[] { 0, 1 }, new double[] { 1, 0 }, new double[] { 0, 0 } };
            //double[][] output = new double[][] { new double[] { 0 }, new double[] { 1 }, new double[] { 1 }, new double[] { 0 } };

            //NeuralNetwork xorNetwork = new NeuralNetwork(new int[] { 2, 2, 1 });
            //xorNetwork.ShowNetwork();
            //int gen = 10000;

            //for (int i = 0; i < gen; i++)
            //{
            //    for (int j = 0; j < input.Length; j++)
            //        xorNetwork.Learning(input[j], output[j]);
            //    if (i % 100 == 0)
            //        xorNetwork.ShowOutput();
            //}

            //xorNetwork.ShowNetwork();
            //var test = xorNetwork.FeedForward(input[0], output[0]);
            //test = xorNetwork.FeedForward(input[1], output[1]);
            //test = xorNetwork.FeedForward(input[2], output[2]);
            //test = xorNetwork.FeedForward(input[3], output[3]);

            Console.WriteLine($"Generations: {gen}");
            Console.Read();
        }
    }
}
