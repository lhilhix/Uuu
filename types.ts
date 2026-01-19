
export enum Role {
  USER = 'user',
  ASSISTANT = 'assistant',
  SYSTEM = 'system'
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  timestamp: Date;
  image?: string; // Base64 encoded image
  video?: string; // URL for generated video
  grounding?: any[]; // Grounding chunks from Google Search/Maps
}

export interface UserProfile {
  username: string;
  avatar: string;
  isAuth: boolean;
  theme: 'light' | 'dark';
}

export enum AppStep {
  SECRET_KEY = 'secret_key',
  USERNAME = 'username',
  CHAT = 'chat'
}
