import React from 'react';
import PredictionForm from "@/components/PredictionForm"


export default function Home(){
  return (
    <main className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8 text-center">Stock/Cryptocurrency Predictor</h1>
      <PredictionForm />
    </main>
  )
}

