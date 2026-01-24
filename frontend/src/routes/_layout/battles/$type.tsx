import { createFileRoute } from "@tanstack/react-router"
import { ArrowLeft, Play, RotateCcw, SkipForward } from "lucide-react"
import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

type BattleType = "tts" | "stt" | "wake" | "intent"

interface BattleData {
  id: string
  type: BattleType
  title: string
  content: string
  candidates: {
    id: string
    name: string
    result?: string
    confidence?: number
    audioUrl?: string
  }[]
}

export const Route = createFileRoute("/_layout/battles/$type")({
  component: Battle,
})

function Battle() {
  const { type } = Route.useParams()
  const [currentBattle, setCurrentBattle] = useState<BattleData | null>(null)
  const [selectedVote, setSelectedVote] = useState<string | null>(null)
  const [isVoting, setIsVoting] = useState(false)
  const [voteSubmitted, setVoteSubmitted] = useState(false)

  // Mock battle data based on type
  const getMockBattleData = (battleType: string): BattleData => {
    switch (battleType) {
      case "tts":
        return {
          id: "tts-1",
          type: "tts",
          title: "🔊 TEXT-TO-SPEECH BATTLE #1",
          content: "The weather today will be sunny with a high of 72 degrees.",
          candidates: [
            {
              id: "A",
              name: "Candidate A",
              audioUrl: "/mock-audio/tts-a.mp3",
            },
            {
              id: "B",
              name: "Candidate B",
              audioUrl: "/mock-audio/tts-b.mp3",
            },
          ],
        }
      case "stt":
        return {
          id: "stt-1",
          type: "stt",
          title: "🎤 SPEECH-TO-TEXT BATTLE #1",
          content: "Set a timer for fifteen minutes",
          candidates: [
            {
              id: "A",
              name: "Candidate A",
              result: "set a timer for 15 minutes",
            },
            {
              id: "B",
              name: "Candidate B",
              result: "sit a timer for fifty minutes",
            },
          ],
        }
      case "wake":
        return {
          id: "wake-1",
          type: "wake",
          title: "👂 WAKE WORD DETECTION BATTLE #1",
          content: "Hey Mycroft",
          candidates: [
            {
              id: "A",
              name: "Detector A",
              result: "TRIGGERED",
              confidence: 89,
            },
            {
              id: "B",
              name: "Detector B",
              result: "NOT TRIGGERED",
              confidence: 12,
            },
          ],
        }
      case "intent":
        return {
          id: "intent-1",
          type: "intent",
          title: "🎯 INTENT CLASSIFICATION BATTLE #1",
          content: "Play some smooth jazz from the 1960s",
          candidates: [
            {
              id: "A",
              name: "Candidate A",
              result: JSON.stringify(
                {
                  intent: "play_music",
                  confidence: 0.94,
                  slots: { genre: "jazz", style: "smooth", decade: "1960s" },
                },
                null,
                2,
              ),
            },
            {
              id: "B",
              name: "Candidate B",
              result: JSON.stringify(
                {
                  intent: "set_timer",
                  confidence: 0.71,
                  slots: { duration: "1960" },
                },
                null,
                2,
              ),
            },
          ],
        }
      default:
        return {
          id: "unknown-1",
          type: "tts",
          title: "UNKNOWN BATTLE",
          content: "Unknown battle type",
          candidates: [],
        }
    }
  }

  // Initialize battle data
  useState(() => {
    setCurrentBattle(getMockBattleData(type))
  })

  const handleVote = async (candidateId: string) => {
    if (isVoting) return

    setIsVoting(true)
    setSelectedVote(candidateId)

    // Simulate API call
    setTimeout(() => {
      setVoteSubmitted(true)
      setIsVoting(false)
    }, 1000)
  }

  const handleNextBattle = () => {
    setVoteSubmitted(false)
    setSelectedVote(null)
    // In real app, this would load next battle
    setCurrentBattle(getMockBattleData(type))
  }

  if (!currentBattle) {
    return <div>Loading battle...</div>
  }

  if (voteSubmitted) {
    return (
      <div className="max-w-2xl mx-auto space-y-6">
        <div className="text-center space-y-4">
          <div className="text-6xl">✅</div>
          <h2 className="text-2xl font-bold">Vote Recorded!</h2>
          <p className="text-muted-foreground">
            You voted for: Candidate {selectedVote}
          </p>
          <p className="text-muted-foreground">Loading next battle...</p>
          <Button onClick={handleNextBattle} className="mt-4">
            Continue to Next Battle
          </Button>
        </div>
      </div>
    )
  }

  const getVoteOptions = () => {
    switch (type) {
      case "tts":
      case "stt":
        return ["A", "B", "Tie"]
      case "wake":
        return ["A", "B"]
      case "intent":
        return ["A", "B", "Tie", "Both Wrong"]
      default:
        return ["A", "B"]
    }
  }

  const getQuestionText = () => {
    switch (type) {
      case "tts":
        return "Which voice sounds better?"
      case "stt":
        return "Which transcription is more accurate?"
      case "wake":
        return "Which detector is correct?"
      case "intent":
        return "Which model understood correctly?"
      default:
        return "Which is better?"
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="sm">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Battle Types
        </Button>
      </div>

      {/* Battle Title */}
      <Card>
        <CardHeader>
          <CardTitle className="text-center">{currentBattle.title}</CardTitle>
        </CardHeader>
      </Card>

      {/* Content */}
      {(type === "tts" || type === "stt") && (
        <Card>
          <CardContent className="pt-6">
            <div className="text-center space-y-2">
              <div className="text-sm font-medium text-muted-foreground">
                {type === "tts"
                  ? "📝 Read this text:"
                  : "🎧 Listen to the original audio:"}
              </div>
              <div className="p-4 bg-muted rounded-lg text-center font-medium">
                {currentBattle.content}
              </div>
              {type === "stt" && (
                <div className="text-sm text-muted-foreground">
                  📄 Reference Text (what was actually said)
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Candidates */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {currentBattle.candidates.map((candidate) => (
          <Card key={candidate.id}>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                🅰️ {candidate.name}
                {type === "wake" && candidate.result && (
                  <Badge
                    variant={
                      candidate.result === "TRIGGERED" ? "default" : "secondary"
                    }
                  >
                    {candidate.result === "TRIGGERED" ? "✅" : "❌"}{" "}
                    {candidate.result}
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {type === "tts" && candidate.audioUrl && (
                <div className="flex items-center justify-center gap-2">
                  <Button variant="outline" size="sm">
                    <Play className="h-4 w-4 mr-2" />
                    Play
                  </Button>
                  <Button variant="outline" size="sm">
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Replay
                  </Button>
                  <div className="flex-1">
                    <Progress value={33} className="h-2" />
                  </div>
                  <span className="text-sm text-muted-foreground">0:03</span>
                </div>
              )}

              {type === "stt" && candidate.result && (
                <div className="p-3 bg-muted rounded text-sm">
                  {candidate.result}
                </div>
              )}

              {type === "wake" && candidate.confidence !== undefined && (
                <div className="text-sm">
                  Confidence: {candidate.confidence}%
                </div>
              )}

              {type === "intent" && candidate.result && (
                <pre className="text-xs bg-muted p-3 rounded overflow-x-auto">
                  {candidate.result}
                </pre>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Voting Section */}
      <Card>
        <CardContent className="pt-6">
          <div className="text-center space-y-4">
            <h3 className="text-lg font-semibold">{getQuestionText()}</h3>

            <div className="flex flex-wrap justify-center gap-2">
              {getVoteOptions().map((option) => (
                <Button
                  key={option}
                  variant={selectedVote === option ? "default" : "outline"}
                  onClick={() => handleVote(option)}
                  disabled={isVoting}
                  className="min-w-[80px]"
                >
                  {isVoting && selectedVote === option
                    ? "Voting..."
                    : `Vote ${option}`}
                </Button>
              ))}
            </div>

            <div className="flex items-center justify-between text-sm text-muted-foreground">
              <Button variant="ghost" size="sm">
                <SkipForward className="h-4 w-4 mr-2" />
                Skip this battle
              </Button>
              <span>Progress: Battle 1 of ∞ | Your votes: 47</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
