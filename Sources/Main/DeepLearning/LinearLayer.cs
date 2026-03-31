using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class LinearLayer : IModule // Zmiana: dodano 'sealed'
    {
        public AutogradNode Weights { get; private set; } 
        public AutogradNode Biases { get; private set; }  
        public bool IsTraining { get; private set; } = true;

        public LinearLayer(int inputSize, int outputSize) 
        {
            var wData = new FastMatrix<double>(inputSize, outputSize); 
            var stdDev = Math.Sqrt(2.0 / inputSize); // Kaiming He Fan-In[cite: 2]

            var wSpan = wData.AsSpan();
            
            for (var i = 0; i < wSpan.Length; i++)
            {
                // Zmiana: Zastosowanie poprawnego rozkładu normalnego Kaiminga za pomocą Box-Mullera
                wSpan[i] = MathUtils.NextGaussian() * stdDev;
            }

            Weights = new AutogradNode(wData, true); 
            Biases = new AutogradNode(new FastMatrix<double>(1, outputSize), true); 
        }
        
        public void Train()
        {
            IsTraining = true;
        }
        public void Eval()
        {
            IsTraining = false;
        }

        public AutogradNode Forward(AutogradNode input) 
        {
            return TensorMath.Linear(input, Weights, Biases); 
        }

        // --- METODY PROSTEGO ZAPISU I ODCZYTU --- 

        public void Save(string path) 
        {
            using var fs = new FileStream(path, FileMode.Create); 
            using var bw = new BinaryWriter(fs); 

            Save(bw); 
        }

        public void Load(string path) 
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Brak pliku wag: {path}"); 

            using var fs = new FileStream(path, FileMode.Open); 
            using var br = new BinaryReader(fs); 

            Load(br); 
        }

        // --- WERSJE NISKOPOZIOMOWE --- 

        public void Save(BinaryWriter bw) 
        {
            bw.Write(Weights.Data.Rows); 
            bw.Write(Weights.Data.Cols); 
            foreach (var val in Weights.Data.AsSpan()) bw.Write(val); 

            bw.Write(Biases.Data.Rows); 
            bw.Write(Biases.Data.Cols); 
            foreach (var val in Biases.Data.AsSpan()) bw.Write(val); 
        }

        public void Load(BinaryReader br) 
        {
            var wRows = br.ReadInt32(); 
            var wCols = br.ReadInt32(); 
            if (wRows != Weights.Data.Rows || wCols != Weights.Data.Cols) 
                throw new Exception("Wymiary wag w pliku nie pasują do architektury!"); 

            var wSpan = Weights.Data.AsSpan(); 
            for (var i = 0; i < wSpan.Length; i++) wSpan[i] = br.ReadDouble(); 

            var bRows = br.ReadInt32(); 
            var bCols = br.ReadInt32(); 
            var bSpan = Biases.Data.AsSpan(); 
            for (var i = 0; i < bSpan.Length; i++) bSpan[i] = br.ReadDouble(); 
        }

        public IEnumerable<AutogradNode> Parameters() 
        {
            yield return Weights; 
            yield return Biases; 
        }

        public void Dispose() 
        {
            Weights?.Dispose(); 
            Biases?.Dispose(); 
        }
    }
}