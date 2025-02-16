"use client"

import { useEffect, useRef } from "react"
import Chart from "chart.js/auto"

interface TrendsGraphProps {
  data: {
    dates: string[]
    buying_trend: number[]
  }
}

export default function TrendsGraph({ data }: TrendsGraphProps) {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)

  useEffect(() => {
    if (chartRef.current) {
      const ctx = chartRef.current.getContext("2d")
      if (ctx) {
        if (chartInstance.current) {
          chartInstance.current.destroy()
        }

        chartInstance.current = new Chart(ctx, {
          type: "line",
          data: {
            labels: data.dates,
            datasets: [
              {
                label: "Buying Trend",
                data: data.buying_trend,
                borderColor: "rgb(75, 192, 192)",
                tension: 0.1,
              },
            ],
          },
          options: {
            responsive: true,
            scales: {
              x: {
                title: {
                  display: true,
                  text: "Date",
                },
              },
              y: {
                title: {
                  display: true,
                  text: "Search Interest",
                },
                beginAtZero: true,
              },
            },
          },
        })
      }
    }

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy()
      }
    }
  }, [data])

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-4">Google Trends Data</h2>
      <canvas ref={chartRef} />
    </div>
  )
}

