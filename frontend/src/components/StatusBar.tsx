import { Database, Zap, Activity } from 'lucide-react'
import type { SystemStatus } from '../types'

interface StatusBarProps {
  status: SystemStatus
}

export default function StatusBar({ status }: StatusBarProps) {
  return (
    <div className="flex items-center gap-4 text-sm">
      <StatusItem
        icon={<Database className="w-4 h-4" />}
        label="Images"
        value={status.total_images.toString()}
        color="text-blue-600"
      />
      <StatusItem
        icon={<Zap className="w-4 h-4" />}
        label="Vectors"
        value={status.total_vectors.toString()}
        color="text-purple-600"
      />
      <StatusItem
        icon={<Activity className="w-4 h-4" />}
        label="Ollama"
        value={status.ollama_connected ? 'Connected' : 'Offline'}
        color={status.ollama_connected ? 'text-green-600' : 'text-red-600'}
      />
    </div>
  )
}

interface StatusItemProps {
  icon: React.ReactNode
  label: string
  value: string
  color: string
}

function StatusItem({ icon, label, value, color }: StatusItemProps) {
  return (
    <div className="flex items-center gap-2">
      <div className={color}>{icon}</div>
      <div>
        <div className="text-xs text-gray-500">{label}</div>
        <div className={`font-semibold ${color}`}>{value}</div>
      </div>
    </div>
  )
}





