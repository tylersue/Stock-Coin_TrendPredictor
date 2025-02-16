import { NextResponse } from "next/server"
import { spawn } from "child_process"
import path from "path"

const PYTHON_SCRIPTS_DIR = path.join(process.cwd(), "python_scripts")

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { ticker } = body
    if (!ticker) {
      return NextResponse.json(
        { error: "Ticker is required." },
        { status: 400 }
      )
    }

    // Wrap our Python process call in a promise for async/await
    const output: string = await new Promise((resolve, reject) => {
      const pythonProcess = spawn("python3", [
        path.join(PYTHON_SCRIPTS_DIR, "StockCoin-TrendPredictor.py"),
        ticker
      ])

      let stdout = ""
      let stderr = ""

      pythonProcess.stdout.on("data", (data) => {
        stdout += data.toString()
      })

      pythonProcess.stderr.on("data", (data) => {
        stderr += data.toString()
      })

      pythonProcess.on("close", (code) => {
        if (code !== 0) {
          return reject(stderr || `Python exited with code ${code}`)
        }
        resolve(stdout)
      })
    })

    try {
      const result = JSON.parse(output)
      return NextResponse.json(result)
    } catch (err) {
      return NextResponse.json(
        { error: "Failed to parse Python output as JSON." },
        { status: 500 }
      )
    }
  } catch (error) {
    return NextResponse.json(
      { error: "Something went wrong with the request." },
      { status: 500 }
    )
  }
}

function runPythonScript(scriptName: string, args: string[]): Promise<any> {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(PYTHON_SCRIPTS_DIR, scriptName)
    const process = spawn("python", [scriptPath, ...args])
    let output = ""

    process.stdout.on("data", (data) => {
      output += data.toString()
    })

    process.stderr.on("data", (data) => {
      console.error(`${scriptName} error:`, data.toString())
    })

    process.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`${scriptName} exited with code ${code}`))
      } else {
        try {
          resolve(JSON.parse(output))
        } catch (error) {
          reject(new Error(`Failed to parse output from ${scriptName}`))
        }
      }
    })
  })
}

