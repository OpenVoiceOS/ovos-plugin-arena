import { FaGithub, FaLinkedinIn } from "react-icons/fa"
import { SiMastodon, SiMatrix } from "react-icons/si"

const socialLinks = [
  { icon: FaGithub, href: "https://github.com/OpenVoiceOS", label: "GitHub" },
  {
    icon: SiMatrix,
    href: "https://matrix.to/#/#openvoiceos:matrix.org",
    label: "Matrix",
  },
  { icon: SiMastodon, href: "https://fosstodon.org/@ovos", label: "Mastodon" },
  {
    icon: FaLinkedinIn,
    href: "https://www.linkedin.com/company/openvoiceos/",
    label: "LinkedIn",
  },
]

export function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="border-t py-4 px-6">
      <div className="flex flex-col items-center justify-between gap-4 sm:flex-row">
        <p className="text-muted-foreground text-sm">
          Open Voice OS - {currentYear}
        </p>
        <div className="flex items-center gap-4">
          {socialLinks.map(({ icon: Icon, href, label }) => (
            <a
              key={label}
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              aria-label={label}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <Icon className="h-5 w-5" />
            </a>
          ))}
        </div>
      </div>
    </footer>
  )
}
