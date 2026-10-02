import { createShapeId, Editor, TLArrowShape, TLShapeId, toRichText } from 'tldraw'
import { z } from 'zod'
import { SchemaBoxShape, schemaHeight } from './SchemaBoxShapeUtil'

const name = z.string().min(1).max(100)
const names = z.array(name).max(30)
export const actionSchema = z.discriminatedUnion('name', [
	z.strictObject({
		name: z.literal('create_schema_box'),
		arguments: z.strictObject({ name, fields: names, methods: names }),
	}),
	z.strictObject({
		name: z.literal('add_property'),
		arguments: z.strictObject({ schema_id: name, property_name: name }),
	}),
	z.strictObject({
		name: z.literal('remove_property'),
		arguments: z.strictObject({ schema_id: name, property_name: name }),
	}),
	z.strictObject({
		name: z.literal('rename_schema'),
		arguments: z.strictObject({ schema_id: name, new_name: name }),
	}),
	z.strictObject({
		name: z.literal('connect_schemas'),
		arguments: z.strictObject({ source_id: name, target_id: name, label: z.string().max(100) }),
	}),
	z.strictObject({
		name: z.literal('no_action'),
		arguments: z.strictObject({
			reason: z.enum(['missing_target', 'ambiguous_target', 'unsupported_request']),
		}),
	}),
])

export type CanvasAction = z.infer<typeof actionSchema>
export interface CanvasContext {
	schemas: { id: string; name: string; properties: string[]; methods: string[] }[]
	selected_ids: string[]
}
export interface VoiceHistory {
	turns: { command: string; action: CanvasAction | null; created_id?: string }[]
	last_created_id: string | null
	last_edited_id: string | null
}

export function canvasContext(editor: Editor): CanvasContext {
	const schemas = editor
		.getCurrentPageShapes()
		.filter((shape): shape is SchemaBoxShape => shape.type === 'voice-schema')
		.sort((a, b) => a.id.localeCompare(b.id))
		.map((shape) => ({
			id: shape.id,
			name: shape.props.name,
			properties: [...shape.props.fields],
			methods: [...shape.props.methods],
		}))
	return {
		schemas,
		selected_ids: editor
			.getSelectedShapeIds()
			.filter((id) => schemas.some((shape) => shape.id === id))
			.sort(),
	}
}

function targetShape(editor: Editor, id: string) {
	const shape = editor.getShape(id as TLShapeId)
	if (!shape || shape.type !== 'voice-schema' || !editor.getCurrentPageShapeIds().has(shape.id)) {
		throw new Error('The target box is no longer on this canvas.')
	}
	return shape
}

export function executeCanvasAction(editor: Editor, raw: unknown, expected: CanvasContext) {
	const action = actionSchema.parse(raw)
	if (JSON.stringify(canvasContext(editor)) !== JSON.stringify(expected)) {
		throw new Error('The canvas changed. Say the command again.')
	}
	if (action.name === 'no_action') return { action, createdId: null, editedId: null }

	let createdId: TLShapeId | null = null
	let editedId: TLShapeId | null = null
	let edit: () => void
	if (action.name === 'create_schema_box') {
		if (expected.schemas.length >= 30) throw new Error('This demo supports up to 30 schema boxes.')
		const { name, fields, methods } = action.arguments
		createdId = editedId = createShapeId()
		const center = editor.getViewportPageBounds().center
		const height = schemaHeight(fields, methods)
		const occupied = editor.getCurrentPageShapes().filter((s) => s.type === 'voice-schema')
		let x = center.x - 140
		const y = center.y - height / 2
		while (occupied.some((s) => Math.abs(s.x - x) < 300 && Math.abs(s.y - y) < height)) x += 400
		edit = () =>
			editor.createShape<SchemaBoxShape>({
				id: createdId!,
				type: 'voice-schema',
				x,
				y,
				props: { w: 280, h: height, name, fields, methods },
			})
	} else if (action.name === 'connect_schemas') {
		const source = targetShape(editor, action.arguments.source_id)
		const target = targetShape(editor, action.arguments.target_id)
		editedId = source.id
		const duplicate = editor.getCurrentPageShapes().some((shape) => {
			if (shape.type !== 'arrow' || shape.meta.voiceLabel !== action.arguments.label) return false
			const bindings = editor.getBindingsFromShape(shape.id, 'arrow')
			return (
				bindings.some((b) => b.props.terminal === 'start' && b.toId === source.id) &&
				bindings.some((b) => b.props.terminal === 'end' && b.toId === target.id)
			)
		})
		edit = () => {
			if (duplicate) return
			const id = createShapeId()
			const origin = editor.getShapePageBounds(source)!.center
			const destination = editor.getShapePageBounds(target)!.center
			editor.createShape<TLArrowShape>({
				id,
				type: 'arrow',
				x: origin.x,
				y: origin.y,
				props: {
					start: { x: 0, y: 0 },
					end: { x: destination.x - origin.x, y: destination.y - origin.y },
					richText: toRichText(action.arguments.label),
					font: 'mono',
					size: 's',
					bend: source.id === target.id ? 100 : 0,
				},
				meta: { voiceLabel: action.arguments.label },
			})
			editor.createBindings([
				{
					type: 'arrow',
					fromId: id,
					toId: source.id,
					props: {
						terminal: 'start',
						normalizedAnchor: { x: 0.5, y: 0.5 },
						isExact: false,
						isPrecise: true,
					},
				},
				{
					type: 'arrow',
					fromId: id,
					toId: target.id,
					props: {
						terminal: 'end',
						normalizedAnchor: { x: 0.5, y: source.id === target.id ? 0.9 : 0.5 },
						isExact: false,
						isPrecise: true,
					},
				},
			])
		}
	} else {
		const shape = targetShape(editor, action.arguments.schema_id)
		editedId = shape.id
		if (action.name === 'rename_schema') {
			edit = () =>
				editor.updateShape<SchemaBoxShape>({
					id: shape.id,
					type: shape.type,
					props: { name: action.arguments.new_name },
				})
		} else {
			const property = action.arguments.property_name
			if (action.name === 'remove_property' && !shape.props.fields.includes(property)) {
				throw new Error('That property no longer exists.')
			}
			const fields = [...shape.props.fields]
			if (action.name === 'remove_property') fields.splice(fields.indexOf(property), 1)
			else if (!fields.includes(property)) fields.push(property)
			if (fields.length > 30) throw new Error('This demo supports up to 30 properties per box.')
			edit = () =>
				editor.updateShape<SchemaBoxShape>({
					id: shape.id,
					type: shape.type,
					props: { fields, h: schemaHeight(fields, shape.props.methods) },
				})
		}
	}
	const mark = editor.markHistoryStoppingPoint('Voice edit')
	try {
		editor.run(() => {
			edit()
			editor.select(editedId!)
		})
	} catch (error) {
		editor.bailToMark(mark)
		throw error
	}
	if (createdId) {
		const bounds = editor.getCurrentPageBounds()
		if (bounds)
			editor.zoomToBounds(bounds, {
				targetZoom: Math.min(editor.getZoomLevel(), 1),
				inset: 64,
				animation: { duration: 200 },
			})
	}
	return { action, createdId, editedId }
}
