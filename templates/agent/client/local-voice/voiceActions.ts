import {
	createShapeId,
	DefaultColorStyle,
	DefaultFillStyle,
	Editor,
	TLArrowShape,
	TLShape,
	TLShapeId,
	toRichText,
} from 'tldraw'
import { z } from 'zod'
import { SchemaBoxShape, schemaHeight } from './SchemaBoxShapeUtil'

const name = z.string().min(1).max(100)
const names = z.array(name).max(30)
const number = z.number().min(-100000).max(100000)
const size = z.number().positive().max(10000)
const color = z.enum([
	'black',
	'grey',
	'light-violet',
	'violet',
	'blue',
	'light-blue',
	'yellow',
	'orange',
	'green',
	'light-green',
	'light-red',
	'red',
	'white',
])
const fill = z.enum(['none', 'semi', 'solid', 'pattern'])
const targets = names
	.min(1)
	.refine((ids) => new Set(ids).size === ids.length, 'Shape IDs must be unique.')
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
	z.strictObject({
		name: z.literal('create_shape'),
		arguments: z.strictObject({
			kind: z.enum([
				'rectangle',
				'ellipse',
				'diamond',
				'triangle',
				'text',
				'note',
				'frame',
				'arrow',
			]),
			text: z.string().max(1000).default(''),
			x: number.nullable().default(null),
			y: number.nullable().default(null),
			width: size.default(160),
			height: size.default(100),
		}),
	}),
	z.strictObject({
		name: z.literal('select_shapes'),
		arguments: z.strictObject({
			shape_ids: names.refine(
				(ids) => new Set(ids).size === ids.length,
				'Shape IDs must be unique.'
			),
		}),
	}),
	z.strictObject({
		name: z.literal('move_shapes'),
		arguments: z.strictObject({ shape_ids: targets, dx: number, dy: number }),
	}),
	z.strictObject({
		name: z.literal('delete_shapes'),
		arguments: z.strictObject({ shape_ids: targets }),
	}),
	z.strictObject({
		name: z.literal('resize_shape'),
		arguments: z.strictObject({ shape_id: name, width: size, height: size }),
	}),
	z.strictObject({
		name: z.literal('set_text'),
		arguments: z.strictObject({ shape_id: name, text: z.string().max(1000) }),
	}),
	z.strictObject({
		name: z.literal('style_shapes'),
		arguments: z.strictObject({
			shape_ids: targets,
			color: color.nullable().default(null),
			fill: fill.nullable().default(null),
			opacity: z.number().min(0).max(1).nullable().default(null),
		}),
	}),
	z.strictObject({
		name: z.literal('arrange_shapes'),
		arguments: z.strictObject({
			shape_ids: targets,
			operation: z.enum([
				'duplicate',
				'group',
				'ungroup',
				'front',
				'back',
				'forward',
				'backward',
				'align_left',
				'align_right',
				'align_top',
				'align_bottom',
				'align_center_horizontal',
				'align_center_vertical',
				'distribute_horizontal',
				'distribute_vertical',
				'flip_horizontal',
				'flip_vertical',
				'stack_horizontal',
				'stack_vertical',
				'pack',
			]),
		}),
	}),
	z.strictObject({
		name: z.literal('canvas_command'),
		arguments: z.strictObject({
			operation: z.enum([
				'undo',
				'redo',
				'select_all',
				'clear_selection',
				'zoom_in',
				'zoom_out',
				'zoom_to_fit',
				'reset_zoom',
			]),
		}),
	}),
	z.strictObject({
		name: z.literal('pan_canvas'),
		arguments: z.strictObject({ dx: number, dy: number }),
	}),
	z.strictObject({
		name: z.literal('add_method'),
		arguments: z.strictObject({ schema_id: name, method_name: name }),
	}),
	z.strictObject({
		name: z.literal('remove_method'),
		arguments: z.strictObject({ schema_id: name, method_name: name }),
	}),
])

export type CanvasAction = z.infer<typeof actionSchema>
interface ShapePosition {
	parent_id: string | null
	rotation: number
	order: number
}
export interface CanvasContext {
	schemas: (ShapePosition & {
		id: string
		name: string
		properties: string[]
		methods: string[]
		x: number
		y: number
		w: number
		h: number
		color: string
		fill: string
		opacity: number
	})[]
	shapes: (ShapePosition & {
		id: string
		name: string
		kind: string
		x: number
		y: number
		w: number
		h: number
		text: string
		color: string
		fill: string
		opacity: number
	})[]
	selected_ids: string[]
	can_undo: boolean
	can_redo: boolean
	camera: { x: number; y: number; z: number }
}
export interface VoiceHistory {
	turns: { command: string; action: CanvasAction | null; created_id?: string }[]
	last_created_id: string | null
	last_edited_id: string | null
}

export function canvasContext(editor: Editor): CanvasContext {
	const pageShapes = editor.getCurrentPageShapes().sort((a, b) => a.id.localeCompare(b.id))
	const order = new Map(
		editor.getCurrentPageShapesSorted().map((shape, index) => [shape.id, index])
	)
	const position = (shape: TLShape): ShapePosition => ({
		parent_id: pageShapes.some((candidate) => candidate.id === shape.parentId)
			? shape.parentId
			: null,
		rotation: shape.rotation,
		order: order.get(shape.id)!,
	})
	const schemas = editor
		.getCurrentPageShapes()
		.filter((shape): shape is SchemaBoxShape => shape.type === 'voice-schema')
		.sort((a, b) => a.id.localeCompare(b.id))
		.map((shape) => {
			const point = editor.getShapePageTransform(shape)!.applyToPoint({ x: 0, y: 0 })
			return {
				...position(shape),
				id: shape.id,
				name: shape.props.name,
				properties: [...shape.props.fields],
				methods: [...shape.props.methods],
				x: point.x,
				y: point.y,
				w: shape.props.w,
				h: shape.props.h,
				color: shape.props.color,
				fill: shape.props.fill,
				opacity: shape.opacity,
			}
		})
	const shapes = pageShapes
		.filter((shape) => shape.type !== 'voice-schema')
		.map((shape) => {
			const bounds = editor.getShapeGeometry(shape).bounds
			const props = shape.props as Record<string, unknown>
			const text = (editor.getShapeUtil(shape).getText(shape) ?? String(props.name ?? '')).slice(
				0,
				1000
			)
			const point = editor.getShapePageTransform(shape)!.applyToPoint({ x: 0, y: 0 })
			return {
				...position(shape),
				id: shape.id,
				name: (text || shape.type).slice(0, 100),
				kind: typeof props.geo === 'string' ? props.geo : shape.type,
				x: point.x,
				y: point.y,
				w: Math.max(1, bounds.w),
				h: Math.max(1, bounds.h),
				text,
				color: typeof props.color === 'string' ? props.color : 'black',
				fill: typeof props.fill === 'string' ? props.fill : 'none',
				opacity: shape.opacity,
			}
		})
	return {
		schemas,
		shapes,
		selected_ids: editor
			.getSelectedShapeIds()
			.filter((id) => pageShapes.some((shape) => shape.id === id))
			.sort(),
		can_undo: editor.getCanUndo(),
		can_redo: editor.getCanRedo(),
		camera: { x: editor.getCamera().x, y: editor.getCamera().y, z: editor.getCamera().z },
	}
}

function pageShape(editor: Editor, id: string) {
	const shape = editor.getShape(id as TLShapeId)
	if (!shape || !editor.getCurrentPageShapeIds().has(shape.id))
		throw new Error('The target shape is no longer on this canvas.')
	return shape
}

function targetShape(editor: Editor, id: string) {
	const shape = pageShape(editor, id)
	if (shape.type !== 'voice-schema') {
		throw new Error('The target box is no longer on this canvas.')
	}
	return shape
}

function rootTargets(editor: Editor, ids: string[]) {
	const shapes = ids.map((id) => pageShape(editor, id))
	const selected = new Set(ids)
	return shapes.filter((shape) => {
		let parent = editor.getShape(shape.parentId as TLShapeId)
		while (parent) {
			if (selected.has(parent.id)) return false
			parent = editor.getShape(parent.parentId as TLShapeId)
		}
		return true
	})
}

function createDrawing(
	editor: Editor,
	id: TLShapeId,
	args: Extract<CanvasAction, { name: 'create_shape' }>['arguments']
) {
	const { kind, text, width, height } = args
	const center = editor.getViewportPageBounds().center
	const position = { id, x: args.x ?? center.x - width / 2, y: args.y ?? center.y - height / 2 }
	switch (kind) {
		case 'rectangle':
		case 'ellipse':
		case 'diamond':
		case 'triangle':
			editor.createShape({
				...position,
				type: 'geo',
				props: { geo: kind, w: width, h: height, richText: toRichText(text) },
			})
			break
		case 'text':
			editor.createShape({
				...position,
				type: 'text',
				props: { w: width, autoSize: false, richText: toRichText(text) },
			})
			break
		case 'note':
			editor.createShape({ ...position, type: 'note', props: { richText: toRichText(text) } })
			break
		case 'frame':
			editor.createShape({ ...position, type: 'frame', props: { w: width, h: height, name: text } })
			break
		case 'arrow':
			editor.createShape({
				...position,
				type: 'arrow',
				props: { start: { x: 0, y: 0 }, end: { x: width, y: height }, richText: toRichText(text) },
			})
	}
}

function arrange(
	editor: Editor,
	shapes: TLShape[],
	operation: Extract<CanvasAction, { name: 'arrange_shapes' }>['arguments']['operation']
) {
	const ids = shapes.map((shape) => shape.id)
	if (operation === 'group' && ids.length < 2)
		throw new Error('Select at least two shapes to group.')
	if (operation === 'ungroup' && shapes.some((shape) => shape.type !== 'group'))
		throw new Error('Only groups can be ungrouped.')
	if (operation.startsWith('distribute') && ids.length < 3)
		throw new Error('Select at least three shapes to distribute.')
	switch (operation) {
		case 'duplicate':
			editor.duplicateShapes(ids, { x: 24, y: 24 })
			break
		case 'group':
			editor.groupShapes(ids, { select: true })
			break
		case 'ungroup':
			editor.ungroupShapes(ids, { select: true })
			break
		case 'front':
			editor.bringToFront(ids)
			break
		case 'back':
			editor.sendToBack(ids)
			break
		case 'forward':
			editor.bringForward(ids)
			break
		case 'backward':
			editor.sendBackward(ids)
			break
		case 'flip_horizontal':
			editor.flipShapes(ids, 'horizontal')
			break
		case 'flip_vertical':
			editor.flipShapes(ids, 'vertical')
			break
		case 'stack_horizontal':
			editor.stackShapes(ids, 'horizontal')
			break
		case 'stack_vertical':
			editor.stackShapes(ids, 'vertical')
			break
		case 'pack':
			editor.packShapes(ids)
			break
		case 'align_left':
			editor.alignShapes(ids, 'left')
			break
		case 'align_right':
			editor.alignShapes(ids, 'right')
			break
		case 'align_top':
			editor.alignShapes(ids, 'top')
			break
		case 'align_bottom':
			editor.alignShapes(ids, 'bottom')
			break
		case 'align_center_horizontal':
			editor.alignShapes(ids, 'center-horizontal')
			break
		case 'align_center_vertical':
			editor.alignShapes(ids, 'center-vertical')
			break
		case 'distribute_horizontal':
			editor.distributeShapes(ids, 'horizontal')
			break
		case 'distribute_vertical':
			editor.distributeShapes(ids, 'vertical')
			break
	}
}

function validateEdit(editor: Editor, action: CanvasAction, expected: CanvasContext) {
	const args = action.arguments
	if ('shape_ids' in args) args.shape_ids.forEach((id) => pageShape(editor, id))
	if ('shape_id' in args) pageShape(editor, args.shape_id)
	if ('schema_id' in args) targetShape(editor, args.schema_id)
	if ('source_id' in args) {
		pageShape(editor, args.source_id)
		pageShape(editor, args.target_id)
	}
	if (
		(action.name === 'create_shape' && expected.shapes.length >= 60) ||
		(action.name === 'create_schema_box' && expected.schemas.length >= 30)
	)
		throw new Error('This edit exceeds the demo canvas limits.')
	if (action.name === 'resize_shape') {
		const shape = pageShape(editor, action.arguments.shape_id)
		const bounds = editor.getShapeGeometry(shape).bounds
		if (!bounds.w || !bounds.h || !editor.getShapeUtil(shape).canResize(shape))
			throw new Error('This shape cannot be resized.')
	}
	if (action.name === 'set_text') {
		const shape = pageShape(editor, action.arguments.shape_id)
		if (shape.type === 'frame' || shape.type === 'voice-schema') {
			if (!action.arguments.text || action.arguments.text.length > 100)
				throw new Error('Box names must contain 1 to 100 characters.')
		} else if (!['geo', 'text', 'note', 'arrow'].includes(shape.type))
			throw new Error('This shape has no editable text.')
	}
	if (action.name === 'arrange_shapes') {
		const shapes = rootTargets(editor, action.arguments.shape_ids)
		const operation = action.arguments.operation
		if (
			(operation === 'group' && shapes.length < 2) ||
			(operation.startsWith('distribute') && shapes.length < 3)
		)
			throw new Error('There are too few shapes for this arrangement.')
		if (operation === 'ungroup' && shapes.some((shape) => shape.type !== 'group'))
			throw new Error('Only groups can be ungrouped.')
		const added =
			operation === 'duplicate'
				? [...editor.getShapeAndDescendantIds(shapes.map((shape) => shape.id))].map(
						(id) => editor.getShape(id)!
					)
				: []
		if (
			expected.schemas.length + added.filter((shape) => shape.type === 'voice-schema').length >
				30 ||
			expected.shapes.length +
				added.filter((shape) => shape.type !== 'voice-schema').length +
				(operation === 'group' ? 1 : 0) >
				60
		)
			throw new Error('This edit exceeds the demo canvas limits.')
	}
	if (
		action.name === 'add_property' ||
		action.name === 'remove_property' ||
		action.name === 'add_method' ||
		action.name === 'remove_method'
	) {
		const shape = targetShape(editor, action.arguments.schema_id)
		const method = action.name === 'add_method' || action.name === 'remove_method'
		const values = method ? shape.props.methods : shape.props.fields
		const value =
			'method_name' in action.arguments
				? action.arguments.method_name
				: action.arguments.property_name
		if (action.name.startsWith('remove') && !values.includes(value))
			throw new Error('That property or method no longer exists.')
		if (action.name.startsWith('add') && values.length >= 30 && !values.includes(value))
			throw new Error('This box is full.')
	}
}

export function executeCanvasAction(editor: Editor, raw: unknown, expected: CanvasContext) {
	const action = actionSchema.parse(raw)
	if (JSON.stringify(canvasContext(editor)) !== JSON.stringify(expected)) {
		throw new Error('The canvas changed. Say the command again.')
	}
	let createdId: TLShapeId | null = null
	let editedId: TLShapeId | null = null
	if (action.name === 'no_action') return { action, createdId, editedId }
	validateEdit(editor, action, expected)
	if (action.name === 'canvas_command') {
		const operation = action.arguments.operation
		if (operation === 'undo' || operation === 'redo') {
			if (operation === 'undo' ? !editor.getCanUndo() : !editor.getCanRedo())
				throw new Error('There is no edit to ' + operation + '.')
			editor[operation]()
		} else {
			editor.run(
				() => {
					switch (operation) {
						case 'select_all':
							editor.selectAll()
							break
						case 'clear_selection':
							editor.selectNone()
							break
						case 'zoom_in':
							editor.zoomIn()
							break
						case 'zoom_out':
							editor.zoomOut()
							break
						case 'zoom_to_fit':
							editor.zoomToFit()
							break
						case 'reset_zoom':
							editor.resetZoom()
							break
					}
				},
				{ history: 'ignore' }
			)
		}
		return { action, createdId, editedId }
	}
	if (action.name === 'select_shapes') {
		const ids = action.arguments.shape_ids.map((id) => pageShape(editor, id).id)
		editor.run(() => editor.select(...ids), { history: 'ignore' })
		return { action, createdId, editedId }
	}
	if (action.name === 'pan_canvas') {
		const { x, y, z } = editor.getCamera()
		editor.setCamera({ x: x - action.arguments.dx, y: y - action.arguments.dy, z })
		return { action, createdId, editedId }
	}
	const before = new Set(editor.getCurrentPageShapeIds())
	const mark = editor.markHistoryStoppingPoint('Voice edit')
	try {
		editor.run(() => {
			switch (action.name) {
				case 'create_shape': {
					if (expected.shapes.length >= 60)
						throw new Error('This demo supports up to 60 drawing shapes.')
					createdId = editedId = createShapeId()
					createDrawing(editor, createdId, action.arguments)
					editor.select(createdId)
					break
				}
				case 'create_schema_box': {
					if (expected.schemas.length >= 30)
						throw new Error('This demo supports up to 30 schema boxes.')
					const { name, fields, methods } = action.arguments
					createdId = editedId = createShapeId()
					const center = editor.getViewportPageBounds().center
					const height = schemaHeight(fields, methods)
					let x = center.x - 140
					const y = center.y - height / 2
					const occupied = expected.schemas
					while (occupied.some((s) => Math.abs(s.x - x) < 300 && Math.abs(s.y - y) < height))
						x += 400
					editor.createShape<SchemaBoxShape>({
						id: createdId,
						type: 'voice-schema',
						x,
						y,
						props: { w: 280, h: height, name, fields, methods },
					})
					editor.select(createdId)
					break
				}
				case 'connect_schemas': {
					const source = pageShape(editor, action.arguments.source_id)
					const target = pageShape(editor, action.arguments.target_id)
					const duplicate = editor.getCurrentPageShapes().some((shape) => {
						if (shape.type !== 'arrow' || shape.meta.voiceLabel !== action.arguments.label)
							return false
						const bindings = editor.getBindingsFromShape(shape.id, 'arrow')
						return (
							bindings.some((b) => b.props.terminal === 'start' && b.toId === source.id) &&
							bindings.some((b) => b.props.terminal === 'end' && b.toId === target.id)
						)
					})
					editedId = source.id
					if (!duplicate) {
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
					editor.select(source.id)
					break
				}
				case 'move_shapes': {
					const shapes = rootTargets(editor, action.arguments.shape_ids)
					const { dx, dy } = action.arguments
					editor.updateShapes(
						shapes.map((shape) => {
							const origin = editor.getShapePageTransform(shape)!.applyToPoint({ x: 0, y: 0 })
							const point = editor.getPointInParentSpace(shape, {
								x: origin.x + dx,
								y: origin.y + dy,
							})
							return { id: shape.id, type: shape.type, x: point.x, y: point.y }
						})
					)
					editedId = shapes[0].id
					editor.select(...shapes.map((shape) => shape.id))
					break
				}
				case 'delete_shapes': {
					const shapes = rootTargets(editor, action.arguments.shape_ids)
					editor.deleteShapes(shapes.map((shape) => shape.id))
					break
				}
				case 'resize_shape': {
					const shape = pageShape(editor, action.arguments.shape_id)
					const bounds = editor.getShapeGeometry(shape).bounds
					if (!bounds.w || !bounds.h || !editor.getShapeUtil(shape).canResize(shape))
						throw new Error('This shape cannot be resized.')
					editor.resizeShape(shape.id, {
						x: action.arguments.width / bounds.w,
						y: action.arguments.height / bounds.h,
					})
					editedId = shape.id
					editor.select(shape.id)
					break
				}
				case 'set_text': {
					const shape = pageShape(editor, action.arguments.shape_id)
					const { text } = action.arguments
					if (shape.type === 'voice-schema' || shape.type === 'frame') {
						if (!text || text.length > 100)
							throw new Error('Box names must contain 1 to 100 characters.')
						editor.updateShape({ id: shape.id, type: shape.type, props: { name: text } })
					} else if (
						shape.type === 'geo' ||
						shape.type === 'text' ||
						shape.type === 'note' ||
						shape.type === 'arrow'
					) {
						editor.updateShape({
							id: shape.id,
							type: shape.type,
							props: { richText: toRichText(text) },
						})
					} else throw new Error('This shape has no editable text.')
					editedId = shape.id
					editor.select(shape.id)
					break
				}
				case 'style_shapes': {
					const shapes = action.arguments.shape_ids.map((id) => pageShape(editor, id))
					editor.select(...shapes.map((shape) => shape.id))
					const { color, fill, opacity } = action.arguments
					if (color !== null) editor.setStyleForSelectedShapes(DefaultColorStyle, color)
					if (fill !== null) editor.setStyleForSelectedShapes(DefaultFillStyle, fill)
					if (opacity !== null) editor.setOpacityForSelectedShapes(opacity)
					editedId = shapes[0].id
					break
				}
				case 'arrange_shapes': {
					const shapes = rootTargets(editor, action.arguments.shape_ids)
					editor.select(...shapes.map((shape) => shape.id))
					arrange(editor, shapes, action.arguments.operation)
					const made = editor.getSelectedShapeIds().find((id) => !before.has(id))
					if (made) createdId = made
					editedId = made ?? editor.getSelectedShapeIds()[0] ?? shapes[0].id
					break
				}
				case 'rename_schema':
				case 'add_property':
				case 'remove_property':
				case 'add_method':
				case 'remove_method': {
					const shape = targetShape(editor, action.arguments.schema_id)
					if (action.name === 'rename_schema') {
						editor.updateShape<SchemaBoxShape>({
							id: shape.id,
							type: shape.type,
							props: { name: action.arguments.new_name },
						})
					} else {
						const method = action.name === 'add_method' || action.name === 'remove_method'
						const value =
							'method_name' in action.arguments
								? action.arguments.method_name
								: action.arguments.property_name
						const fields = [...shape.props.fields]
						const methods = [...shape.props.methods]
						const values = method ? methods : fields
						if (action.name.startsWith('remove')) {
							if (!values.includes(value))
								throw new Error('That property or method no longer exists.')
							values.splice(values.indexOf(value), 1)
						} else if (!values.includes(value)) values.push(value)
						if (values.length > 30)
							throw new Error('This demo supports up to 30 properties or methods per box.')
						editor.updateShape<SchemaBoxShape>({
							id: shape.id,
							type: shape.type,
							props: { fields, methods, h: schemaHeight(fields, methods) },
						})
					}
					editedId = shape.id
					editor.select(shape.id)
				}
			}
			if (
				editor.getCurrentPageShapes().filter((shape) => shape.type !== 'voice-schema').length >
					60 ||
				editor.getCurrentPageShapes().filter((shape) => shape.type === 'voice-schema').length > 30
			)
				throw new Error('This edit exceeds the demo canvas limits.')
		})
	} catch (error) {
		editor.bailToMark(mark)
		throw error
	}
	return { action, createdId, editedId }
}
