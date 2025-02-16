"use client"

import type React from "react"

import { useState } from "react"
import PredictionResult from "./PredictionResult"
import TrendsGraph from "./TrendsGraph"

export default function PredictionForm() {
  const [ticker, setTicker] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [predictionData, setPredictionData] = useState<any>(null)
  const [trendsData, setTrendsData] = useState<any>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setPredictionData(null)
    setTrendsData(null)

    try {
      const response = await fetch("/Users/tylersue/Downloads/stock-predictor/python_scripts/StockCoin-TrendPredictor.py", {
        
        method: "POST",

        
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ ticker }),
      })

      // if (!response.ok) {
      //  throw new Error("Failed to fetch prediction data")
      // }

      const data = await response.json()
      setPredictionData(data.prediction)
      setTrendsData(data.trends)
    } catch (err) {
   //   setError("An error occurred while fetching the prediction data.")
    } finally {
      setLoading(false)
    }
  }


  return (
    <div>
      <form onSubmit={handleSubmit} className="mb-8">
        <div className="flex items-center">
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            placeholder="Enter stock ticker (e.g., AAPL, BTC-USD)"

            className="flex-grow p-2 border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          />
          <button
            type="submit"
            className="bg-blue-500 text-white p-2 rounded-r-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={loading}
          >
            {loading ? "Loading..." : "Predict"}
          </button>
        </div>
      </form>

      {error && <p className="text-red-500 mb-4">{error}</p>}

      {predictionData && <PredictionResult data={predictionData} />}
      {trendsData && <TrendsGraph data={trendsData} />}
    </div>
  )
}

