import { createRoot } from 'react-dom/client'
import { Tldraw, Editor } from 'tldraw'
import 'tldraw/tldraw.css'
import { SchemaBoxShapeUtil } from '../client/local-voice/SchemaBoxShapeUtil'
import { canvasContext, executeCanvasAction } from '../client/local-voice/voiceActions'

declare global {
	interface Window {
		editor: Editor
		voice: { context: typeof canvasContext; execute: typeof executeCanvasAction }
	}
}

createRoot(document.getElementById('root')!).render(
	<div style={{ position: 'fixed', inset: 0 }}>
		<Tldraw
			shapeUtils={[SchemaBoxShapeUtil]}
			onMount={(editor) => {
				window.editor = editor
				window.voice = { context: canvasContext, execute: executeCanvasAction }
			}}
		/>
	</div>
)
