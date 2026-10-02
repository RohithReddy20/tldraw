import { FormEvent, useEffect, useRef, useState } from 'react'
import { Editor, Tldraw, TLUiOverrides } from 'tldraw'
import { z } from 'zod'
import { SchemaBoxShapeUtil } from './SchemaBoxShapeUtil'
import { actionSchema, canvasContext, executeCanvasAction, VoiceHistory } from './voiceActions'
import './voice.css'

const shapeUtils = [SchemaBoxShapeUtil]
const overrides: TLUiOverrides = { translations: { en: { 'tool.voice-schema': 'Schema box' } } }
const service = 'http://127.0.0.1:8097'
const responseSchema = z.object({ command: z.string().min(1).max(2000), action: actionSchema })

function audioBase64(blob: Blob): Promise<string> {
	if (blob.size > 6_000_000)
		return Promise.reject(new Error('Recording is too large. Try a shorter command.'))
	return new Promise((resolve, reject) => {
		const reader = new FileReader()
		reader.onload = () => resolve((reader.result as string).split(',')[1])
		reader.onerror = () => reject(new Error('Could not read the recording.'))
		reader.readAsDataURL(blob)
	})
}

export function LocalVoiceApp() {
	const [editor, setEditor] = useState<Editor | null>(null)
	const [command, setCommand] = useState('')
	const [lastCommand, setLastCommand] = useState('')
	const [status, setStatus] = useState('Ready')
	const [error, setError] = useState('')
	const [online, setOnline] = useState(false)
	const [busy, setBusy] = useState(false)
	const [recording, setRecording] = useState(false)
	const [seconds, setSeconds] = useState(0)
	const history = useRef<VoiceHistory>({ turns: [], last_created_id: null, last_edited_id: null })
	const recorder = useRef<MediaRecorder | null>(null)
	const stream = useRef<MediaStream | null>(null)
	const recordingTimeout = useRef<ReturnType<typeof setTimeout> | null>(null)
	const request = useRef<AbortController | null>(null)
	const alive = useRef(true)
	const working = useRef(false)

	useEffect(() => {
		alive.current = true
		const controller = new AbortController()
		const check = () => {
			fetch(`${service}/health`, { signal: controller.signal })
				.then((response) => {
					if (alive.current) setOnline(response.ok)
				})
				.catch(() => {
					if (alive.current) setOnline(false)
				})
		}
		check()
		const poll = setInterval(check, 10_000)
		return () => {
			alive.current = false
			controller.abort()
			request.current?.abort()
			clearInterval(poll)
			if (recordingTimeout.current) clearTimeout(recordingTimeout.current)
			if (recorder.current?.state === 'recording') recorder.current.stop()
			stream.current?.getTracks().forEach((track) => track.stop())
		}
	}, [])

	useEffect(() => {
		if (!recording) return
		const timer = setInterval(() => setSeconds((value) => value + 1), 1000)
		return () => clearInterval(timer)
	}, [recording])

	async function send(input: { command: string } | { audio: string }) {
		if (!editor || working.current) return
		working.current = true
		setBusy(true)
		setError('')
		setStatus('Drawing')
		const snapshot = canvasContext(editor)
		const controller = new AbortController()
		request.current = controller
		let recognizedCommand = 'command' in input ? input.command : null
		try {
			const response = await fetch(`${service}/command`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ ...input, canvas: snapshot, history: history.current }),
				signal: controller.signal,
			})
			const body = await response.json()
			if (!response.ok) {
				const failure = z.object({ error: z.string() }).safeParse(body)
				throw new Error(failure.success ? failure.data.error : 'The command could not be applied.')
			}
			const result = responseSchema.parse(body)
			recognizedCommand = result.command
			if (!alive.current) return
			const outcome = executeCanvasAction(editor, result.action, snapshot)
			history.current = {
				turns: [
					...history.current.turns,
					{
						command: result.command,
						action: outcome.action,
						...(outcome.createdId ? { created_id: outcome.createdId } : {}),
					},
				].slice(-3),
				last_created_id: outcome.createdId ?? history.current.last_created_id,
				last_edited_id: outcome.editedId ?? history.current.last_edited_id,
			}
			setOnline(true)
			setLastCommand(result.command)
			setCommand('')
			setStatus(outcome.action.name === 'no_action' ? 'No edit' : 'Applied')
		} catch (failure) {
			if (alive.current && !controller.signal.aborted) {
				if (recognizedCommand) {
					history.current.turns = [
						...history.current.turns,
						{ command: recognizedCommand, action: null },
					].slice(-3)
				}
				setStatus('Try again')
				setError(
					failure instanceof TypeError
						? 'Start the local voice service to apply commands.'
						: failure instanceof Error
							? failure.message
							: 'The command could not be applied.'
				)
			}
		} finally {
			working.current = false
			if (alive.current) setBusy(false)
		}
	}

	async function toggleRecording() {
		if (recorder.current?.state === 'recording') {
			recorder.current.stop()
			return
		}
		if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
			setError('Microphone recording is unavailable in this browser. Type a command below.')
			return
		}
		setError('')
		setBusy(true)
		try {
			const microphone = await navigator.mediaDevices.getUserMedia({ audio: true })
			if (!alive.current) {
				microphone.getTracks().forEach((track) => track.stop())
				return
			}
			stream.current = microphone
			const mimeType = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/ogg;codecs=opus'].find(
				(type) => MediaRecorder.isTypeSupported(type)
			)
			const capture = new MediaRecorder(microphone, mimeType ? { mimeType } : {})
			recorder.current = capture
			const chunks: Blob[] = []
			let failed = false
			capture.ondataavailable = (event) => {
				if (event.data.size) chunks.push(event.data)
			}
			capture.onstop = async () => {
				microphone.getTracks().forEach((track) => track.stop())
				if (recordingTimeout.current) clearTimeout(recordingTimeout.current)
				if (!alive.current) return
				setRecording(false)
				if (failed) {
					setStatus('Try again')
					return
				}
				setBusy(true)
				try {
					await send({ audio: await audioBase64(new Blob(chunks, { type: capture.mimeType })) })
				} catch (failure) {
					if (alive.current) {
						setError(failure instanceof Error ? failure.message : 'Recording failed.')
						setStatus('Try again')
					}
				} finally {
					if (alive.current) setBusy(false)
				}
			}
			capture.onerror = () => {
				failed = true
				if (recordingTimeout.current) clearTimeout(recordingTimeout.current)
				microphone.getTracks().forEach((track) => track.stop())
				if (alive.current) {
					setRecording(false)
					setError('Recording failed. Try again.')
				}
			}
			capture.start()
			setSeconds(0)
			setRecording(true)
			setStatus('Listening')
			recordingTimeout.current = setTimeout(() => {
				if (capture.state === 'recording') capture.stop()
			}, 60_000)
		} catch (failure) {
			stream.current?.getTracks().forEach((track) => track.stop())
			setError(failure instanceof Error ? failure.message : 'Could not access the microphone.')
		} finally {
			if (alive.current) setBusy(false)
		}
	}

	function submit(event: FormEvent) {
		event.preventDefault()
		if (command.trim()) void send({ command: command.trim() })
	}

	return (
		<main className="local-voice-app">
			<header className="voice-header">
				<h1>
					Voice canvas<span> / schema studio</span>
				</h1>
				<div className="voice-service" data-online={online}>
					<span />
					{online ? 'Local service ready' : 'Local service offline'}
				</div>
			</header>
			<div className="voice-canvas">
				<Tldraw
					persistenceKey="local-voice-canvas-v1"
					shapeUtils={shapeUtils}
					overrides={overrides}
					onMount={setEditor}
				/>
			</div>
			<section className="voice-console" aria-label="Voice commands">
				<div className="voice-console-top">
					<button
						className="voice-record"
						data-recording={recording}
						disabled={busy || !editor}
						onClick={() => void toggleRecording()}
						aria-pressed={recording}
					>
						<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
							<rect
								x="9"
								y="2"
								width="6"
								height="13"
								rx="3"
								fill="none"
								stroke="currentColor"
								strokeWidth="1.8"
							/>
							<path
								d="M6 10v2a6 6 0 0 0 12 0v-2M12 18v4M8 22h8"
								fill="none"
								stroke="currentColor"
								strokeWidth="1.8"
							/>
						</svg>
						{recording ? 'Finish recording' : 'Speak'}
					</button>
					<div className="voice-status" role="status" aria-live="polite">
						<span data-active={recording || busy} />
						{status}
						{recording && ` · ${seconds}s`}
					</div>
					<span className="voice-silent">Speak → canvas</span>
				</div>
				<form className="voice-form" onSubmit={submit}>
					<input
						aria-label="Canvas command"
						value={command}
						maxLength={2000}
						onChange={(event) => setCommand(event.target.value)}
						placeholder="Create a User schema with name, class, subjects…"
						disabled={busy || recording}
					/>
					<button
						type="submit"
						disabled={busy || recording || !command.trim() || !editor}
						aria-label="Apply command"
					>
						↵
					</button>
				</form>
				{error ? (
					<p className="voice-error" role="alert">
						{error}
					</p>
				) : (
					<p className="voice-transcript">
						{lastCommand || 'Create a box. Select it. Keep refining.'}
					</p>
				)}
			</section>
		</main>
	)
}
