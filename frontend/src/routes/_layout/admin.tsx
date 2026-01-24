import { createFileRoute } from "@tanstack/react-router"
import { BarChart3, Database, Edit, Package, Plus } from "lucide-react"
import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Dialog, DialogContent, DialogTrigger } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import useAuth from "@/hooks/useAuth"

type ModalityType = "tts" | "stt" | "wake" | "intent"

interface Competitor {
  id: string
  name: string
  modality: ModalityType
  displayName?: string
  config: Record<string, any>
  elo: number
  active: boolean
  battles: number
}

export const Route = createFileRoute("/_layout/admin")({
  component: Admin,
  head: () => ({
    meta: [
      {
        title: "Admin Panel - OVOS Plugin Arena",
      },
    ],
  }),
})

function Admin() {
  const { user: currentUser } = useAuth()
  const [activeTab, setActiveTab] = useState("competitors")
  const [showAddDialog, setShowAddDialog] = useState(false)

  // Mock data
  const [competitors, setCompetitors] = useState<Competitor[]>([
    {
      id: "1",
      name: "ovos-tts-plugin-piper",
      modality: "tts",
      displayName: "Piper EN-US Lessac",
      config: { voice: "en_US-lessac-medium", quality: "medium", speed: 1.0 },
      elo: 1523,
      active: true,
      battles: 412,
    },
    {
      id: "2",
      name: "ovos-tts-plugin-coqui",
      modality: "tts",
      displayName: "Coqui TTS Jenny",
      config: { voice: "jenny", quality: "high", speed: 1.0 },
      elo: 1489,
      active: true,
      battles: 387,
    },
    {
      id: "3",
      name: "ovos-stt-plugin-whisper",
      modality: "stt",
      displayName: "Whisper Large V3",
      config: { model: "large-v3", language: "en" },
      elo: 1589,
      active: true,
      battles: 445,
    },
  ])

  const getCompetitorsByModality = (modality: ModalityType) => {
    return competitors.filter((c) => c.modality === modality)
  }

  const handleAddCompetitor = (
    newCompetitor: Omit<Competitor, "id" | "elo" | "battles">,
  ) => {
    const competitor: Competitor = {
      ...newCompetitor,
      id: Date.now().toString(),
      elo: 1200,
      battles: 0,
    }
    setCompetitors([...competitors, competitor])
    setShowAddDialog(false)
  }

  const toggleCompetitorStatus = (id: string) => {
    setCompetitors(
      competitors.map((c) => (c.id === id ? { ...c, active: !c.active } : c)),
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">🛠️ ADMIN PANEL</h1>
        <p className="text-muted-foreground">
          Welcome back, {currentUser?.full_name || currentUser?.email}!
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="competitors" className="flex items-center gap-2">
            <Package className="h-4 w-4" />
            Competitors
          </TabsTrigger>
          <TabsTrigger value="analytics" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Analytics
          </TabsTrigger>
          <TabsTrigger value="data" className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            Data
          </TabsTrigger>
        </TabsList>

        <TabsContent value="competitors" className="space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>📦 REGISTERED COMPETITORS</CardTitle>
                <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
                  <DialogTrigger asChild>
                    <Button>
                      <Plus className="h-4 w-4 mr-2" />
                      Add New Competitor
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-2xl">
                    <AddCompetitorDialog onAdd={handleAddCompetitor} />
                  </DialogContent>
                </Dialog>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* TTS Competitors */}
              <div>
                <h3 className="font-semibold mb-2">
                  🔊 TTS Competitors ({getCompetitorsByModality("tts").length}{" "}
                  active)
                </h3>
                <div className="space-y-2">
                  {getCompetitorsByModality("tts").map((competitor) => (
                    <div
                      key={competitor.id}
                      className="flex items-center justify-between p-3 border rounded"
                    >
                      <div>
                        <div className="font-medium">
                          {competitor.displayName || competitor.name}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {competitor.name}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge
                          variant={competitor.active ? "default" : "secondary"}
                        >
                          {competitor.active ? "Active" : "Inactive"}
                        </Badge>
                        <Button variant="outline" size="sm">
                          <Edit className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => toggleCompetitorStatus(competitor.id)}
                        >
                          {competitor.active ? "Deactivate" : "Activate"}
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* STT Competitors */}
              <div>
                <h3 className="font-semibold mb-2">
                  🎤 STT Competitors ({getCompetitorsByModality("stt").length}{" "}
                  active)
                </h3>
                <div className="space-y-2">
                  {getCompetitorsByModality("stt").map((competitor) => (
                    <div
                      key={competitor.id}
                      className="flex items-center justify-between p-3 border rounded"
                    >
                      <div>
                        <div className="font-medium">
                          {competitor.displayName || competitor.name}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {competitor.name}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge
                          variant={competitor.active ? "default" : "secondary"}
                        >
                          {competitor.active ? "Active" : "Inactive"}
                        </Badge>
                        <Button variant="outline" size="sm">
                          <Edit className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => toggleCompetitorStatus(competitor.id)}
                        >
                          {competitor.active ? "Deactivate" : "Activate"}
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Similar sections for Wake Word and Intent */}
              <div>
                <h3 className="font-semibold mb-2">
                  👂 Wake Word Competitors (
                  {getCompetitorsByModality("wake").length} active)
                </h3>
                <div className="text-sm text-muted-foreground">
                  No wake word competitors registered yet.
                </div>
              </div>

              <div>
                <h3 className="font-semibold mb-2">
                  🎯 Intent Competitors (
                  {getCompetitorsByModality("intent").length} active)
                </h3>
                <div className="text-sm text-muted-foreground">
                  No intent competitors registered yet.
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>📊 SYSTEM ANALYTICS</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="font-semibold mb-4">
                  Overall Stats (Last 30 days):
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold">2,847</div>
                    <div className="text-sm text-muted-foreground">
                      Total Battles
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold">2,654</div>
                    <div className="text-sm text-muted-foreground">
                      Total Votes
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold">127</div>
                    <div className="text-sm text-muted-foreground">
                      Active Users
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold">20.9</div>
                    <div className="text-sm text-muted-foreground">
                      Avg. Votes/User
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="font-semibold mb-4">By Modality:</h3>
                <div className="space-y-2">
                  <div className="flex justify-between items-center p-3 border rounded">
                    <span>🔊 TTS</span>
                    <span className="font-medium">1,245 battles (47%)</span>
                  </div>
                  <div className="flex justify-between items-center p-3 border rounded">
                    <span>🎤 STT</span>
                    <span className="font-medium">687 battles (26%)</span>
                  </div>
                  <div className="flex justify-between items-center p-3 border rounded">
                    <span>👂 Wake Word</span>
                    <span className="font-medium">412 battles (15%)</span>
                  </div>
                  <div className="flex justify-between items-center p-3 border rounded">
                    <span>🎯 Intent</span>
                    <span className="font-medium">503 battles (19%)</span>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="font-semibold mb-4">Top Contributors:</h3>
                <div className="space-y-2">
                  <div className="flex justify-between items-center p-2">
                    <span>1. john.doe@email.com</span>
                    <Badge>234 votes</Badge>
                  </div>
                  <div className="flex justify-between items-center p-2">
                    <span>2. sarah.smith@email.com</span>
                    <Badge>189 votes</Badge>
                  </div>
                  <div className="flex justify-between items-center p-2">
                    <span>3. alex.chen@email.com</span>
                    <Badge>156 votes</Badge>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="data" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>🗂️ SAMPLE DATA MANAGEMENT</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold">
                    📝 TTS Text Samples (234 samples)
                  </h3>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm">
                      ➕ Add Sample
                    </Button>
                    <Button variant="outline" size="sm">
                      📤 Import CSV
                    </Button>
                    <Button variant="outline" size="sm">
                      Export
                    </Button>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="p-3 border rounded">
                    <div className="text-sm">
                      "The weather today will be sunny with a high of 72
                      degrees."
                    </div>
                    <div className="flex gap-2 mt-2">
                      <Button variant="outline" size="sm">
                        Edit
                      </Button>
                      <Button variant="outline" size="sm">
                        Delete
                      </Button>
                    </div>
                  </div>
                  <div className="p-3 border rounded">
                    <div className="text-sm">
                      "Please set a timer for five minutes"
                    </div>
                    <div className="flex gap-2 mt-2">
                      <Button variant="outline" size="sm">
                        Edit
                      </Button>
                      <Button variant="outline" size="sm">
                        Delete
                      </Button>
                    </div>
                  </div>
                </div>
                <div className="text-center mt-4">
                  <Button variant="outline">[Show all...]</Button>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold">
                    🎤 STT Audio Files (156 files)
                  </h3>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm">
                      ➕ Upload Audio
                    </Button>
                    <Button variant="outline" size="sm">
                      📤 Import Batch
                    </Button>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="p-3 border rounded">
                    <div className="font-medium">common-voice-001.wav</div>
                    <div className="text-sm text-muted-foreground">
                      Reference: "Set an alarm for seven AM"
                    </div>
                    <div className="flex gap-2 mt-2">
                      <Button variant="outline" size="sm">
                        ▶️ Play
                      </Button>
                      <Button variant="outline" size="sm">
                        Edit
                      </Button>
                      <Button variant="outline" size="sm">
                        Delete
                      </Button>
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="font-semibold mb-4">
                  👂 Wake Word Audio (89 positive, 112 negative)
                </h3>
                <Button variant="outline" size="sm">
                  ➕ Upload Audio
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

function AddCompetitorDialog({
  onAdd,
}: {
  onAdd: (competitor: Omit<Competitor, "id" | "elo" | "battles">) => void
}) {
  const [formData, setFormData] = useState({
    name: "",
    modality: "tts" as ModalityType,
    displayName: "",
    config: "{}",
    active: true,
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const config = JSON.parse(formData.config)
      onAdd({
        name: formData.name,
        modality: formData.modality,
        displayName: formData.displayName || undefined,
        config,
        active: formData.active,
      })
    } catch (_error) {
      alert("Invalid JSON configuration")
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <Label htmlFor="modality">Modality</Label>
        <Select
          value={formData.modality}
          onValueChange={(value: ModalityType) =>
            setFormData({ ...formData, modality: value })
          }
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="tts">🔊 TTS</SelectItem>
            <SelectItem value="stt">🎤 STT</SelectItem>
            <SelectItem value="wake">👂 Wake Word</SelectItem>
            <SelectItem value="intent">🎯 Intent</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div>
        <Label htmlFor="name">Plugin Name</Label>
        <Input
          id="name"
          value={formData.name}
          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
          placeholder="ovos-tts-plugin-piper"
          required
        />
      </div>

      <div>
        <Label htmlFor="displayName">Display Name (Optional)</Label>
        <Input
          id="displayName"
          value={formData.displayName}
          onChange={(e) =>
            setFormData({ ...formData, displayName: e.target.value })
          }
          placeholder="Piper Amy (Fast)"
        />
      </div>

      <div>
        <Label htmlFor="config">Configuration (JSON)</Label>
        <Textarea
          id="config"
          value={formData.config}
          onChange={(e) => setFormData({ ...formData, config: e.target.value })}
          placeholder='{"voice": "en_US-amy-medium", "quality": "medium", "speed": 1.2}'
          rows={4}
          required
        />
      </div>

      <div className="flex items-center space-x-2">
        <Checkbox
          id="active"
          checked={formData.active}
          onCheckedChange={(checked) =>
            setFormData({ ...formData, active: checked as boolean })
          }
        />
        <Label htmlFor="active">Activate immediately</Label>
      </div>

      <div className="flex items-center space-x-2">
        <Checkbox defaultChecked />
        <Label>Include in random battles</Label>
      </div>

      <div className="flex justify-end gap-2">
        <Button type="button" variant="outline">
          Cancel
        </Button>
        <Button type="submit">✅ Register</Button>
      </div>
    </form>
  )
}
