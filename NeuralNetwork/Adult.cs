using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    class Adult
    {
        private List<long[]> dataSet;
        private string rawData;
        const string path = "Dataset.data";
        public double[][] NormalizedDataSet { get; private set; }
        public double[][] TestSet { get; private set; }
        public double[][] LearningTest { get; private set; }
        
        public Adult()
        {
            this.LoadData(Properties.Resource.Adult);
            this.ProcessData();
        }

        private void SetData(double[][] data)
        {
            Random rnd = new Random();
            var randomizedArray = data.OrderBy(x => rnd.Next()).ToArray();
            this.TestSet = randomizedArray.Take((int) (randomizedArray.Length * 0.3)).ToArray();
            this.LearningTest = randomizedArray.Skip((int) (randomizedArray.Length * 0.3)).ToArray();
        }

        private void ProcessData()
        {
            string[] workclass = { "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked" };
            string[] education = { "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool" };
            string[] martial_status = { "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse" };
            string[] occupation = { "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces" };
            string[] relationship = { "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried" };
            string[] race = { "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black" };
            string[] sex = { "Female", "Male" };
            string[] native_country = { "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands" };
            string[] result = { ">50K", "<=50K" };

        string[][] stringData =
        {
            null,
            workclass,
            null,
            education,
            null,
            martial_status,
            occupation,
            relationship,
            race,
            sex,
            null,
            null,
            null,
            native_country,
            result
        };

        var splitData = this.rawData.Split('\n').Select(x => x.Split(' ').ToArray()).ToArray();
            this.dataSet = new List<long[]>();
            foreach (var row in splitData)
            {
                if (!row.Contains("?"))
                {
                    var tmp = new long[15];
                    for (int i = 0; i < row.Length; i++)
                    {
                        if (stringData[i] == null)
                        {
                            tmp[i] = long.Parse(row[i]);
                        }
                        else
                        {
                            tmp[i] = Array.IndexOf(stringData[i], row[i]);
                        }
                    }

                    this.dataSet.Add(tmp);
                }
            }
        }

        public double[][] NormalizeData()
        {
            var normalizedData = new double[this.dataSet.Count][];
            for (int i = 0; i < this.dataSet.Count; i++)
            {
                normalizedData[i] = new double[this.dataSet[i].Length];
            }

            var columnValues = new List<long[]>();
            for (int i = 0; i < this.dataSet[0].Length; i++)
            {
                columnValues.Add(this.dataSet.Select(x => x[i]).ToArray());
            }

            for(int i = 0; i < this.dataSet[0].Length; i++)
            {
                var min = columnValues[i].Min();
                var max = columnValues[i].Max();

                for (int j = 0; j < this.dataSet.Count; j++)
                {
                    normalizedData[j][i] = MinMax(this.dataSet[j][i], min, max);
                }
            }

            this.NormalizedDataSet = normalizedData;
            SetData(normalizedData);
            return normalizedData;
        }

        private double MinMax(double value, double min, double max) => (value - min) / (max - min);

        private void LoadData(byte[] path)
        {
            try
            {
                this.rawData = Encoding.UTF8.GetString(path);
            }
            catch (Exception)
            {
                Console.WriteLine("Failed to open the data file");
            }
        }
    }
}
