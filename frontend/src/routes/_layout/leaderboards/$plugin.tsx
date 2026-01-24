import { createFileRoute, useNavigate } from "@tanstack/react-router"
import { ArrowLeft, Play, Swords } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

type CategoryType = "tts" | "stt" | "wake" | "intent"

interface PluginDetails {
  name: string
  displayName: string
  rank: number
  category: CategoryType
  config: Record<string, any>
  stats: {
    currentElo: number
    peakElo: number
    peakDate: string
    totalBattles: number
    wins: number
    losses: number
    ties: number
    winRate: number
  }
  eloHistory: Array<{ date: string; elo: number }>
}

export const Route = createFileRoute("/_layout/leaderboards/$plugin")({
  component: PluginDetails,
  validateSearch: (search) => ({
    category: (search.category as CategoryType) || "tts",
  }),
})

function PluginDetails() {
  const { plugin } = Route.useParams()
  const { category } = Route.useSearch()
  const navigate = useNavigate()

  // Mock plugin data
  const getPluginData = (
    pluginName: string,
    cat: CategoryType,
  ): PluginDetails => {
    const baseData = {
      name: pluginName,
      displayName: pluginName,
      rank: 1,
      category: cat,
      config: {
        voice: "en_US-lessac-medium",
        quality: "medium",
        speed: 1.0,
      },
      stats: {
        currentElo: 1523,
        peakElo: 1547,
        peakDate: "Jan 12, 2026",
        totalBattles: 412,
        wins: 313,
        losses: 67,
        ties: 32,
        winRate: 76,
      },
      eloHistory: [
        { date: "Dec 20", elo: 1400 },
        { date: "Dec 25", elo: 1420 },
        { date: "Jan 1", elo: 1480 },
        { date: "Jan 10", elo: 1520 },
        { date: "Jan 18", elo: 1523 },
      ],
    }

    // Customize based on plugin
    if (pluginName === "coqui-tts-jenny") {
      baseData.rank = 2
      baseData.stats.currentElo = 1489
      baseData.stats.totalBattles = 387
      baseData.stats.wins = 275
      baseData.stats.winRate = 71
      baseData.config = {
        voice: "jenny",
        quality: "high",
        speed: 1.0,
      }
    }

    return baseData
  }

  const pluginData = getPluginData(plugin, category)

  const handleBattleClick = () => {
    navigate({
      to: "/battles/$type",
      params: { type: category },
      search: {},
    })
  }

  const getRankEmoji = (rank: number) => {
    switch (rank) {
      case 1:
        return "🥇"
      case 2:
        return "🥈"
      case 3:
        return "🥉"
      default:
        return rank.toString()
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button
          variant="ghost"
          onClick={() =>
            navigate({
              to: "/leaderboards",
              search: { category },
            })
          }
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to {category.toUpperCase()} Leaderboard
        </Button>
      </div>

      {/* Plugin Details */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            📦 PLUGIN DETAILS
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Basic Info */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold mb-2">Name:</h3>
              <p className="text-sm bg-muted p-2 rounded">{pluginData.name}</p>
            </div>
            <div>
              <h3 className="font-semibold mb-2">Rank:</h3>
              <Badge variant="secondary" className="text-lg px-3 py-1">
                {getRankEmoji(pluginData.rank)} #{pluginData.rank} in{" "}
                {category.toUpperCase()} category
              </Badge>
            </div>
          </div>

          {/* Configuration */}
          <div>
            <h3 className="font-semibold mb-2">⚙️ Configuration:</h3>
            <pre className="bg-muted p-4 rounded text-sm overflow-x-auto">
              {JSON.stringify(pluginData.config, null, 2)}
            </pre>
          </div>

          {/* Performance Stats */}
          <div>
            <h3 className="font-semibold mb-4">📊 Performance Stats:</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold">
                  {pluginData.stats.currentElo}
                </div>
                <div className="text-sm text-muted-foreground">Current ELO</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold">
                  {pluginData.stats.peakElo}
                </div>
                <div className="text-sm text-muted-foreground">
                  Peak ELO ({pluginData.stats.peakDate})
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold">
                  {pluginData.stats.totalBattles}
                </div>
                <div className="text-sm text-muted-foreground">
                  Total Battles
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold">
                  {pluginData.stats.winRate}%
                </div>
                <div className="text-sm text-muted-foreground">Win Rate</div>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-4 mt-4">
              <div className="text-center">
                <div className="text-xl font-bold text-green-600">
                  {pluginData.stats.wins}
                </div>
                <div className="text-sm text-muted-foreground">Wins</div>
              </div>
              <div className="text-center">
                <div className="text-xl font-bold text-red-600">
                  {pluginData.stats.losses}
                </div>
                <div className="text-sm text-muted-foreground">Losses</div>
              </div>
              <div className="text-center">
                <div className="text-xl font-bold text-yellow-600">
                  {pluginData.stats.ties}
                </div>
                <div className="text-sm text-muted-foreground">Ties</div>
              </div>
            </div>
          </div>

          {/* ELO History Chart (Mock) */}
          <div>
            <h3 className="font-semibold mb-2">
              📈 ELO History (Last 30 days):
            </h3>
            <div className="bg-muted p-8 rounded text-center">
              <div className="text-muted-foreground">
                [Mock ELO Chart - Would show line graph here]
              </div>
              <div className="mt-4 text-sm">Current trend: 📈 Rising</div>
            </div>
          </div>

          {/* Sample Audio (for TTS) */}
          {category === "tts" && (
            <div>
              <h3 className="font-semibold mb-2">🔊 Listen to Sample:</h3>
              <Button variant="outline">
                <Play className="h-4 w-4 mr-2" />
                Play sample generated by this config
              </Button>
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-4">
            <Button onClick={handleBattleClick}>
              <Swords className="h-4 w-4 mr-2" />
              Battle this plugin
            </Button>
            <Button variant="outline">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
