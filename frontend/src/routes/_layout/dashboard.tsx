import { createFileRoute, useNavigate } from "@tanstack/react-router"
import { Ear, Mic, Target, Volume2 } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import useAuth from "@/hooks/useAuth"

export const Route = createFileRoute("/_layout/dashboard")({
  component: Dashboard,
  head: () => ({
    meta: [
      {
        title: "OVOS Plugin Arena - Dashboard",
      },
    ],
  }),
})

function Dashboard() {
  const { user: currentUser } = useAuth()
  const navigate = useNavigate()

  const battleTypes = [
    {
      id: "tts",
      icon: Volume2,
      title: "Text-to-Speech",
      description: "Text-to-Speech",
      votes: 47,
      color: "bg-blue-500",
    },
    {
      id: "stt",
      icon: Mic,
      title: "Speech-to-Text",
      description: "Speech-to-Text",
      votes: 23,
      color: "bg-green-500",
    },
    {
      id: "wake",
      icon: Ear,
      title: "Wake Detection",
      description: "Wake Detection",
      votes: 12,
      color: "bg-purple-500",
    },
    {
      id: "intent",
      icon: Target,
      title: "Classification",
      description: "Intent Classification",
      votes: 31,
      color: "bg-orange-500",
    },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">OVOS Plugin Arena</h1>
        <p className="text-muted-foreground">
          Welcome back, {currentUser?.full_name || currentUser?.email}!
        </p>
      </div>

      <div>
        <h2 className="text-xl font-semibold mb-4">Choose a Battle Type:</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {battleTypes.map((type) => {
            const Icon = type.icon
            return (
              <Card
                key={type.id}
                className="cursor-pointer hover:shadow-lg transition-shadow"
              >
                <CardHeader className="pb-3">
                  <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg ${type.color}`}>
                      <Icon className="h-6 w-6 text-white" />
                    </div>
                    <div>
                      <CardTitle className="text-lg">{type.title}</CardTitle>
                      <CardDescription>{type.description}</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <Badge variant="secondary">Your votes: {type.votes}</Badge>
                    <Button
                      onClick={() =>
                        navigate({
                          to: "/battles/$type",
                          params: { type: type.id },
                          search: {},
                        })
                      }
                    >
                      Start Battle
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Your Stats Today</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold">113</div>
              <div className="text-sm text-muted-foreground">Total Votes</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold">5</div>
              <div className="text-sm text-muted-foreground">Day Streak</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold">Top 10%</div>
              <div className="text-sm text-muted-foreground">Rank</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
