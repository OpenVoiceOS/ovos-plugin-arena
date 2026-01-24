import { createFileRoute, useNavigate } from "@tanstack/react-router"
import { Ear, Mic, Target, Volume2 } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

type CategoryType = "tts" | "stt" | "wake" | "intent"

interface LeaderboardEntry {
  rank: string
  name: string
  elo: number
  battles: number
  winRate: string
}

export const Route = createFileRoute("/_layout/leaderboards/")({
  component: Leaderboards,
  validateSearch: (search) => ({
    category: (search.category as CategoryType) || "tts",
  }),
})

function Leaderboards() {
  const { category } = Route.useSearch()
  const navigate = useNavigate()
  const [selectedCategory, setSelectedCategory] =
    useState<CategoryType>(category)

  const categories = [
    {
      id: "tts" as const,
      icon: Volume2,
      title: "TTS",
      fullTitle: "Text-to-Speech",
    },
    {
      id: "stt" as const,
      icon: Mic,
      title: "STT",
      fullTitle: "Speech-to-Text",
    },
    { id: "wake" as const, icon: Ear, title: "Wake", fullTitle: "Wake Word" },
    {
      id: "intent" as const,
      icon: Target,
      title: "Intent",
      fullTitle: "Intent Classification",
    },
  ]

  // Mock leaderboard data
  const getLeaderboardData = (cat: CategoryType): LeaderboardEntry[] => {
    switch (cat) {
      case "tts":
        return [
          {
            rank: "🥇 1",
            name: "piper-en-us-lessac",
            elo: 1523,
            battles: 412,
            winRate: "76%",
          },
          {
            rank: "🥈 2",
            name: "coqui-tts-jenny",
            elo: 1489,
            battles: 387,
            winRate: "71%",
          },
          {
            rank: "🥉 3",
            name: "mimic3-en-us-low",
            elo: 1456,
            battles: 356,
            winRate: "68%",
          },
          {
            rank: "4",
            name: "espeak-ng-default",
            elo: 1398,
            battles: 298,
            winRate: "63%",
          },
          {
            rank: "5",
            name: "mary-tts-us-male",
            elo: 1367,
            battles: 276,
            winRate: "59%",
          },
          {
            rank: "6",
            name: "festival-default",
            elo: 1334,
            battles: 245,
            winRate: "57%",
          },
          {
            rank: "7",
            name: "pico-tts-en-us",
            elo: 1298,
            battles: 203,
            winRate: "54%",
          },
          {
            rank: "8",
            name: "flite-cmu-us-rms",
            elo: 1267,
            battles: 189,
            winRate: "51%",
          },
        ]
      case "stt":
        return [
          {
            rank: "🥇 1",
            name: "whisper-large-v3",
            elo: 1589,
            battles: 445,
            winRate: "78%",
          },
          {
            rank: "🥈 2",
            name: "coqui-stt-en",
            elo: 1542,
            battles: 398,
            winRate: "74%",
          },
          {
            rank: "🥉 3",
            name: "vosk-model-en",
            elo: 1498,
            battles: 367,
            winRate: "69%",
          },
        ]
      case "wake":
        return [
          {
            rank: "🥇 1",
            name: "precise-hey-mycroft",
            elo: 1423,
            battles: 289,
            winRate: "72%",
          },
          {
            rank: "🥈 2",
            name: "snowboy-hey-mycroft",
            elo: 1389,
            battles: 256,
            winRate: "68%",
          },
        ]
      case "intent":
        return [
          {
            rank: "🥇 1",
            name: "padatious-default",
            elo: 1467,
            battles: 334,
            winRate: "73%",
          },
          {
            rank: "🥈 2",
            name: "adapt-parser",
            elo: 1421,
            battles: 298,
            winRate: "69%",
          },
        ]
      default:
        return []
    }
  }

  const handleCategoryChange = (newCategory: CategoryType) => {
    setSelectedCategory(newCategory)
    navigate({
      to: "/leaderboards",
      search: { category: newCategory },
      replace: true,
    })
  }

  const handlePluginClick = (pluginName: string) => {
    navigate({
      to: "/leaderboards/$plugin",
      params: { plugin: pluginName },
      search: { category: selectedCategory },
    })
  }

  const currentCategory = categories.find((c) => c.id === selectedCategory)
  const leaderboardData = getLeaderboardData(selectedCategory)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <h1 className="text-3xl font-bold">🏆 Leaderboards</h1>
      </div>

      {/* Category Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Choose a category:</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {categories.map((cat) => {
              const Icon = cat.icon
              return (
                <Button
                  key={cat.id}
                  variant={selectedCategory === cat.id ? "default" : "outline"}
                  onClick={() => handleCategoryChange(cat.id)}
                  className="flex items-center gap-2"
                >
                  <Icon className="h-4 w-4" />
                  {cat.title}
                </Button>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Leaderboard */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {currentCategory && <currentCategory.icon className="h-5 w-5" />}
            {currentCategory?.fullTitle.toUpperCase()} LEADERBOARD
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="grid grid-cols-5 gap-4 font-semibold text-sm border-b pb-2">
              <div>Rank</div>
              <div>Plugin Name</div>
              <div>ELO</div>
              <div>Battles</div>
              <div>Win%</div>
            </div>
            {leaderboardData.map((item) => (
              <div
                key={item.rank}
                className="grid grid-cols-5 gap-4 text-sm py-2 border-b last:border-b-0 hover:bg-muted/50 cursor-pointer"
                onClick={() => handlePluginClick(item.name)}
              >
                <div>{item.rank}</div>
                <div className="font-medium">{item.name}</div>
                <div>{item.elo}</div>
                <div>{item.battles}</div>
                <div>{item.winRate}</div>
              </div>
            ))}
          </div>
          <div className="mt-4 text-center">
            <Button variant="outline">Load More</Button>
          </div>
          <div className="mt-4 text-center text-sm text-muted-foreground">
            Click any plugin to see details →
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
