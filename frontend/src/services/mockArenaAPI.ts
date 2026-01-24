// Mock API services for OVOS Plugin Arena
// These return sample data instead of making real API calls

export interface BattleData {
  id: string
  type: "tts" | "stt" | "wake" | "intent"
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

export interface LeaderboardEntry {
  rank: string
  name: string
  elo: number
  battles: number
  winRate: string
}

export interface PluginDetails {
  name: string
  displayName: string
  rank: number
  category: "tts" | "stt" | "wake" | "intent"
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

export interface Competitor {
  id: string
  name: string
  modality: "tts" | "stt" | "wake" | "intent"
  displayName?: string
  config: Record<string, any>
  elo: number
  active: boolean
  battles: number
}

export interface UserStats {
  totalVotes: number
  streak: number
  rank: string
  votesByType: Record<string, number>
}

export interface SystemAnalytics {
  totalBattles: number
  totalVotes: number
  activeUsers: number
  avgVotesPerUser: number
  battlesByModality: Record<string, number>
  topContributors: Array<{ email: string; votes: number }>
}

// Mock delay to simulate API calls
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

export class MockArenaAPI {
  // Get next battle for a specific type
  static async getBattle(
    type: "tts" | "stt" | "wake" | "intent",
  ): Promise<BattleData> {
    await delay(500)

    switch (type) {
      case "tts":
        return {
          id: `tts-${Date.now()}`,
          type: "tts",
          title: "🔊 TEXT-TO-SPEECH BATTLE",
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
          id: `stt-${Date.now()}`,
          type: "stt",
          title: "🎤 SPEECH-TO-TEXT BATTLE",
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
          id: `wake-${Date.now()}`,
          type: "wake",
          title: "👂 WAKE WORD DETECTION BATTLE",
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
          id: `intent-${Date.now()}`,
          type: "intent",
          title: "🎯 INTENT CLASSIFICATION BATTLE",
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
    }
  }

  // Submit a vote for a battle
  static async submitVote(
    _battleId: string,
    candidateId: string,
  ): Promise<{ success: boolean; message: string }> {
    await delay(800)
    return {
      success: true,
      message: `Vote recorded for Candidate ${candidateId}`,
    }
  }

  // Get leaderboard for a category
  static async getLeaderboard(
    category: "tts" | "stt" | "wake" | "intent",
  ): Promise<LeaderboardEntry[]> {
    await delay(300)

    switch (category) {
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
    }
  }

  // Get plugin details
  static async getPluginDetails(
    pluginName: string,
    category: "tts" | "stt" | "wake" | "intent",
  ): Promise<PluginDetails> {
    await delay(400)

    const baseData = {
      name: pluginName,
      displayName: pluginName,
      rank: 1,
      category: category,
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

  // Get user stats
  static async getUserStats(): Promise<UserStats> {
    await delay(200)
    return {
      totalVotes: 113,
      streak: 5,
      rank: "Top 10% contributor",
      votesByType: {
        tts: 47,
        stt: 23,
        wake: 12,
        intent: 31,
      },
    }
  }

  // Get all competitors (admin)
  static async getCompetitors(): Promise<Competitor[]> {
    await delay(300)
    return [
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
    ]
  }

  // Add new competitor (admin)
  static async addCompetitor(
    competitor: Omit<Competitor, "id" | "elo" | "battles">,
  ): Promise<Competitor> {
    await delay(600)
    return {
      ...competitor,
      id: Date.now().toString(),
      elo: 1200,
      battles: 0,
    }
  }

  // Update competitor status (admin)
  static async updateCompetitorStatus(
    _id: string,
    _active: boolean,
  ): Promise<{ success: boolean }> {
    await delay(400)
    return { success: true }
  }

  // Get system analytics (admin)
  static async getSystemAnalytics(): Promise<SystemAnalytics> {
    await delay(500)
    return {
      totalBattles: 2847,
      totalVotes: 2654,
      activeUsers: 127,
      avgVotesPerUser: 20.9,
      battlesByModality: {
        tts: 1245,
        stt: 687,
        wake: 412,
        intent: 503,
      },
      topContributors: [
        { email: "john.doe@email.com", votes: 234 },
        { email: "sarah.smith@email.com", votes: 189 },
        { email: "alex.chen@email.com", votes: 156 },
      ],
    }
  }

  // Get sample data (admin)
  static async getSampleData(type: "tts" | "stt" | "wake"): Promise<any[]> {
    await delay(300)

    switch (type) {
      case "tts":
        return [
          {
            id: 1,
            text: "The weather today will be sunny with a high of 72 degrees.",
          },
          { id: 2, text: "Please set a timer for five minutes" },
          { id: 3, text: "What time is it?" },
        ]
      case "stt":
        return [
          {
            id: 1,
            filename: "common-voice-001.wav",
            reference: "Set an alarm for seven AM",
          },
          {
            id: 2,
            filename: "common-voice-002.wav",
            reference: "Play some music",
          },
        ]
      case "wake":
        return [
          { id: 1, filename: "positive-001.wav", type: "positive" },
          { id: 2, filename: "negative-001.wav", type: "negative" },
        ]
      default:
        return []
    }
  }
}
