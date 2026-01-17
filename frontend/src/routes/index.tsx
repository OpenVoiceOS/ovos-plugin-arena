import { createFileRoute, Link as RouterLink } from "@tanstack/react-router"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Trophy, Activity, Database, Vote, BarChart3, GitCompare, ArrowRight } from "lucide-react"

export const Route = createFileRoute("/")({
  component: Landing,
  head: () => ({
    meta: [
      {
        title: "OVOS Plugin Arena",
      },
    ],
  }),
})

function Landing() {
  return (
    <div className="min-h-screen bg-black text-zinc-100 selection:bg-white selection:text-black font-sans">
      {/* "Startup" Grid Background */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>
        <div className="absolute left-0 right-0 top-0 -z-10 m-auto h-[310px] w-[310px] rounded-full bg-white opacity-[0.03] blur-[100px]"></div>
      </div>

      <div className="relative z-10">
        {/* Navbar Placeholder (implies minimalism) */}
        <nav className="fixed top-0 w-full z-50 px-6 py-4 border-b border-white/5 bg-black/50 backdrop-blur-xl">
          <div className="max-w-6xl mx-auto flex justify-between items-center">
            <div className="text-sm font-bold tracking-tighter">OVOS ARENA</div>
            <div className="flex gap-4 text-xs font-medium text-zinc-400">
              <RouterLink to="/login" className="hover:text-white transition-colors">Sign In</RouterLink>
            </div>
          </div>
        </nav>

        {/* Hero Section */}
        <section className="px-6 pt-40 pb-32">
          <div className="max-w-4xl mx-auto text-center space-y-10">
            {/* Minimal Badge */}
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-white/10 bg-white/5 backdrop-blur-sm">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-zinc-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-zinc-200"></span>
              </span>
              <span className="text-xs font-medium text-zinc-300 tracking-wide uppercase">Beta 2026</span>
            </div>

            {/* Main Title */}
            <div className="space-y-4">
              <h1 className="text-6xl md:text-8xl font-medium tracking-tighter text-transparent bg-clip-text bg-gradient-to-b from-white via-white to-zinc-500">
                PLUGIN ARENA
              </h1>
              
              <p className="text-xl md:text-2xl text-zinc-400 max-w-2xl mx-auto font-light leading-relaxed tracking-tight">
                The objective benchmark for voice intelligence. <br className="hidden md:block" />
                Evaluate. Battle. Rank.
              </p>
            </div>

            {/* CTA Buttons - High Contrast */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center pt-8">
              <Button 
                size="lg" 
                asChild 
                className="bg-white text-black hover:bg-zinc-200 rounded-full px-8 h-12 font-medium tracking-tight transition-all"
              >
                <RouterLink to="/signup">
                  Enter Arena
                </RouterLink>
              </Button>
              <Button 
                size="lg" 
                variant="outline" 
                asChild
                className="bg-transparent border-zinc-800 text-zinc-300 hover:text-white hover:bg-zinc-900 rounded-full px-8 h-12 font-medium tracking-tight"
              >
                <RouterLink to="/login" className="flex items-center gap-2">
                  Live Data <ArrowRight className="w-4 h-4" />
                </RouterLink>
              </Button>
            </div>

            {/* Stats - Minimal Numbers */}
            <div className="pt-20 border-t border-white/5 grid grid-cols-3 gap-8 max-w-lg mx-auto">
              {[
                { label: "Datasets", value: "12+" },
                { label: "Battles", value: "24/7" },
                { label: "Metric", value: "ELO" },
              ].map((stat, i) => (
                <div key={i} className="text-center">
                  <div className="text-2xl font-semibold tracking-tighter text-white">{stat.value}</div>
                  <div className="text-xs text-zinc-500 uppercase tracking-widest mt-1">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Features / Bento Grid */}
        <section className="px-4 py-24 border-t border-white/5 bg-zinc-950/50">
          <div className="max-w-6xl mx-auto">
            <div className="mb-16">
              <h2 className="text-3xl font-medium tracking-tighter mb-4">Architecture</h2>
              <p className="text-zinc-500 text-lg">Engineered for transparency.</p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              <FeatureCard 
                icon={<Activity />}
                title="Continuous Eval"
                desc="Rankings update continuously based on real-time battles and automated dataset pipelines."
              />
              <FeatureCard 
                icon={<BarChart3 />}
                title="ELO System"
                desc="Chess-inspired ranking distribution ensures statistical significance in plugin comparisons."
              />
              <FeatureCard 
                icon={<GitCompare />}
                title="Benchmarks"
                desc="Reproducible testing environments against immutable, version-controlled reference sets."
              />
              <FeatureCard 
                icon={<Vote />}
                title="Community Alignment"
                desc="RLHF-style voting mechanism allowing users to shape the trajectory of plugin adoption."
              />
              <FeatureCard 
                icon={<Trophy />}
                title="Live Leaderboard"
                desc="Zero-latency updates on global rankings, trending plugins, and performance outliers."
              />
              <FeatureCard 
                icon={<Database />}
                title="Open Data"
                desc="All metrics, battle logs, and evaluation methodologies are open source and verifiable."
              />
            </div>
          </div>
        </section>

        {/* "Why" Section - Text Heavy, Editorial */}
        <section className="px-6 py-32">
          <div className="max-w-3xl mx-auto space-y-8">
            <h2 className="text-4xl font-medium tracking-tighter">The Signal in the Noise.</h2>
            <div className="space-y-6 text-lg text-zinc-400 font-light leading-relaxed">
              <p>
                The ecosystem is flooded with plugins. Quantity has outpaced quality. 
                <span className="text-zinc-200 font-normal"> The Arena exists to solve discovery through rigor.</span>
              </p>
              <p>
                By combining automated reference checks with community-driven preference testing, 
                we create a self-correcting hierarchy of capability. No marketing fluff. 
                Just raw performance data.
              </p>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="px-6 py-12 border-t border-white/5">
          <div className="max-w-6xl mx-auto flex justify-between items-center text-xs text-zinc-600">
            <p>© 2026 OVOS. Open Source Intelligence.</p>
            <div className="flex gap-6">
              <span className="cursor-pointer hover:text-zinc-400">GitHub</span>
              <span className="cursor-pointer hover:text-zinc-400">Twitter</span>
              <span className="cursor-pointer hover:text-zinc-400">Docs</span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  )
}

// Reusable Minimal Card Component
function FeatureCard({ icon, title, desc }: { icon: any, title: string, desc: string }) {
  return (
    <Card className="bg-black border border-white/10 hover:border-white/20 transition-colors duration-300 rounded-xl group">
      <CardHeader>
        <div className="w-10 h-10 rounded-lg bg-zinc-900 flex items-center justify-center mb-2 text-zinc-400 group-hover:text-white transition-colors">
          {icon}
        </div>
        <CardTitle className="text-lg font-medium text-zinc-200 tracking-tight">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-zinc-500 leading-relaxed font-normal">
          {desc}
        </p>
      </CardContent>
    </Card>
  )
}