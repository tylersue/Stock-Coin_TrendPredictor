interface PredictionResultProps {
  data: {
    next_day_movement: string
    future_prices: number[]
    model_score: number
  }
}

export default function PredictionResult({ data }: PredictionResultProps) {
  const { next_day_movement, future_prices, model_score } = data

  return (
    <div className="bg-white p-6 rounded-lg shadow-md mb-8">
      <h2 className="text-2xl font-bold mb-4">Prediction Results</h2>
      <p className="mb-2">
        <span className="font-semibold">Next day's movement:</span>{" "}
        <span className={next_day_movement === "Up" ? "text-green-500" : "text-red-500"}>{next_day_movement}</span>
      </p>
      <p className="mb-4">
        <span className="font-semibold">Model R² score:</span> {model_score.toFixed(4)}
      </p>
      <h3 className="text-xl font-semibold mb-2">Predicted prices for the next 10 days:</h3>
      <ul className="list-disc list-inside">
        {future_prices.map((price, index) => (
          <li key={index}>
            Day {index + 1}: ${price.toFixed(2)}
          </li>
        ))}
      </ul>
    </div>
  )
}

