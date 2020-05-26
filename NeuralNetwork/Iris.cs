using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    class Iris
    {
        double[][] Data;
        double[][] output;
        List<string> irisName = new List<string>();
        int lenght;

        public double[][] TestData { get; private set; }
        public double[][] LearningData { get; private set; }
        public double[][] TestOutput { get; private set; }
        public double[][] LearningOutput { get; private set; }

        public Iris(byte[] file)
        {
            LoadData(file);
            Normalize();
            SetData();
        }

        void LoadData(byte[] file)
        {
            try
            {
                string rawData = System.Text.Encoding.UTF8.GetString(file);
                //Wywołanie metody pobierającej dane do tablicy
                GetData(rawData.Split('\n'));
            }
            catch (Exception)
            {
                Console.WriteLine("Failed to open the data file");
            }
        }
        //Pobieranie danych do tablicy 
        void GetData(string[] splitData)
        {
            lenght = splitData.Length;
            Data = new double[4][] { new double[lenght], new double[lenght], new double[lenght], new double[lenght]};
            output = new double[lenght][];
            for (int i = 0; i < lenght; i++)
            {
                var tmpSplitData = splitData[i].Split(',');
                Data[0][i] = Double.Parse(tmpSplitData[0], new CultureInfo("en"));
                Data[1][i] = Double.Parse(tmpSplitData[1], new CultureInfo("en"));
                Data[2][i] = Double.Parse(tmpSplitData[2], new CultureInfo("en"));
                Data[3][i] = Double.Parse(tmpSplitData[3], new CultureInfo("en"));

                //Zapisywanie nazwy jako liczby
                output[i] = new double[3];
                if (irisName.Contains(tmpSplitData[4]))
                {
                    output[i][irisName.IndexOf(tmpSplitData[4])] = 1;
                }
                else
                {
                    irisName.Add(tmpSplitData[4]);
                    output[i][irisName.Count-1] = 1;
                }
            }
        }
        
        //Normalizacja danych
        void Normalize()
        {
            Data[0] = MinMax(Data[0]);
            Data[1] = MinMax(Data[1]);
            Data[2] = MinMax(Data[2]);
            Data[3] = MinMax(Data[3]);
        }

        double[] MinMax(double[] data)
        {
            double max = data.Max();
            double min = data.Min();

            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] == max)
                    data[i] = 1;
                else if (data[i] == min)
                    data[i] = 0;
                else
                    data[i] = (data[i] - min) / (max - min);
            }
            return data;
        }

        //Tasowanie danych
        void ShuffleData()
        {
            for (int t = 0; t < lenght; t++)
            {
                Random R = new Random();
                int r = R.Next(t, lenght);
                foreach(var column in Data)
                {
                    var tmp = column[t];
                    column[t] = column[r];
                    column[r] = tmp;
                }
                var tmpO = output[t];
                output[t] = output[r];
                output[r] = tmpO;
            }
        }

        //Podział danych na testujące i uczące
        void SetData()
        {
            ShuffleData();
            LearningData = new double[(int)Math.Ceiling(Data[0].Length * 0.7)][];
            for (int i = 0; i < (int)Data[0].Length*0.7; i++)//150
            {
                LearningData[i] = new double[Data.Length];
                for (int j = 0; j < Data.Length; j++)//5
                {
                    LearningData[i][j] = Data[j][i];
                }
            }
            TestData = new double[(int)Math.Ceiling(Data[0].Length * 0.3)][];
            for (int i = 0; i < (int)Math.Ceiling(Data[0].Length * 0.3); i++)//150
            {
                TestData[i] = new double[Data.Length];
                for (int j = 0; j < Data.Length; j++)//5
                {
                    TestData[i][j] = Data[j][i];
                }
            }

            LearningOutput = output.Take((int)(lenght * 0.7)).ToArray();
            TestOutput = output.Skip((int)(lenght * 0.7)).ToArray();
        }

        public string GetName(double[] output)
        {
            int i;
            for (i = 0; i < output.Length; i++)
            {
                if (output[i] == 1)
                    return irisName[i];
            }
            return $"Nazwa o indexie {i} nie istnieje";
        }

        public void ShowData()
        {
            for (int i = 0; i < Data[0].Length; i++)
            {
                Console.WriteLine($"{Data[0][i]}, {Data[1][i]}, {Data[2][i]}, {Data[3][i]}, {Data[4][i]} = {GetName(output[i])}");
            }
        }
    }
}
