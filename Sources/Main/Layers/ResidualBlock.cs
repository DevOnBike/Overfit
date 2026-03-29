namespace DevOnBike.Overfit.Layers
{
    /// <summary>
    /// Klasyczny Blok ResNet (Residual Block).
    /// Architektura: Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm -> (+) Dodanie Wejścia -> ReLU
    /// </summary>
    public class ResidualBlock
    {
        private readonly LinearLayer _linear1;
        private readonly BatchNorm1D _bn1;
        private readonly LinearLayer _linear2;
        private readonly BatchNorm1D _bn2;

        public ResidualBlock(int hiddenSize)
        {
            // W bloku resztkowym wymiar wejściowy musi być równy wyjściowemu,
            // aby można było dodać do siebie macierze (Shape match)
            _linear1 = new LinearLayer(hiddenSize, hiddenSize);
            _bn1 = new BatchNorm1D(hiddenSize);
            
            _linear2 = new LinearLayer(hiddenSize, hiddenSize);
            _bn2 = new BatchNorm1D(hiddenSize);
        }

        public Tensor Forward(Tensor input, bool isTraining)
        {
            // --- ŚCIEŻKA GŁÓWNA (F(X)) ---
            // 1. Pierwsza warstwa + stabilizacja + aktywacja
            var out1 = _linear1.Forward(input);
            var bn1Out = _bn1.Forward(out1, isTraining);
            var a1 = TensorMath.ReLU(bn1Out);

            // 2. Druga warstwa + stabilizacja (bez aktywacji!)
            var out2 = _linear2.Forward(a1);
            var bn2Out = _bn2.Forward(out2, isTraining);

            // --- POŁĄCZENIE RESZTKOWE (SKIP CONNECTION) ---
            // F(X) + X
            // Tutaj dzieje się magia Autogradu. Węzeł 'Add' ma dwoje dzieci.
            var added = TensorMath.Add(bn2Out, input);

            // 3. Ostateczna aktywacja po dodaniu
            return TensorMath.ReLU(added);
        }

        public IEnumerable<Tensor> Parameters()
        {
            return _linear1.Parameters()
                .Concat(_bn1.Parameters())
                .Concat(_linear2.Parameters())
                .Concat(_bn2.Parameters());
        }

        public void Save(string pathPrefix)
        {
            _linear1.Save($"{pathPrefix}_l1.bin");
            _bn1.Save($"{pathPrefix}_bn1.bin");
            _linear2.Save($"{pathPrefix}_l2.bin");
            _bn2.Save($"{pathPrefix}_bn2.bin");
        }

        public void Load(string pathPrefix)
        {
            _linear1.Load($"{pathPrefix}_l1.bin");
            _bn1.Load($"{pathPrefix}_bn1.bin");
            _linear2.Load($"{pathPrefix}_l2.bin");
            _bn2.Load($"{pathPrefix}_bn2.bin");
        }
    }
}