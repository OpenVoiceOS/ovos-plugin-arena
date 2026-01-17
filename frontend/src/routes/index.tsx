import { createFileRoute, Link as RouterLink } from "@tanstack/react-router"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Mic, Brain, Zap, Users } from "lucide-react"

export const Route = createFileRoute("/")({
  component: Landing,
  head: () => ({
    meta: [
      {
        title: "Open Voice OS Beta",
      },
    ],
  }),
})

function Landing() {
  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="py-20 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-5xl font-bold mb-6">
            Open Voice OS <Badge className="ml-2">Beta</Badge>
          </h1>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Experience the future of voice interaction with our AI-powered operating system.
            Seamless conversations, intelligent responses, and natural voice commands.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" asChild>
              <RouterLink to="/signup">Get Started</RouterLink>
            </Button>
            <Button size="lg" variant="outline" asChild>
              <RouterLink to="/login">Log In</RouterLink>
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 px-4 bg-muted/50">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">What is Open Voice OS?</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            <Card>
              <CardHeader className="text-center">
                <Mic className="w-12 h-12 mx-auto mb-4 text-primary" />
                <CardTitle>Natural Voice Interaction</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center">
                  Communicate naturally with your devices using advanced voice recognition and synthesis.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="text-center">
                <Brain className="w-12 h-12 mx-auto mb-4 text-primary" />
                <CardTitle>AI-Powered Intelligence</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center">
                  Leverage cutting-edge AI to understand context, learn preferences, and provide intelligent responses.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="text-center">
                <Zap className="w-12 h-12 mx-auto mb-4 text-primary" />
                <CardTitle>Lightning Fast</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center">
                  Experience instant responses and seamless interactions with our optimized voice processing engine.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="text-center">
                <Users className="w-12 h-12 mx-auto mb-4 text-primary" />
                <CardTitle>Community Driven</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center">
                  Join our growing community of developers and users shaping the future of voice technology.
                </CardDescription>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4">
        <div className="max-w-2xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6">Ready to Get Started?</h2>
          <p className="text-lg text-muted-foreground mb-8">
            Join our beta program and be among the first to experience the next generation of voice interaction.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" asChild>
              <RouterLink to="/signup">Sign Up for Beta</RouterLink>
            </Button>
            <Button size="lg" variant="outline" asChild>
              <RouterLink to="/login">Existing User? Log In</RouterLink>
            </Button>
          </div>
        </div>
      </section>
    </div>
  )
}