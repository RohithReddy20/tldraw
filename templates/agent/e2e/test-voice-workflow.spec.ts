import { readFileSync } from 'node:fs'
import { expect, test } from '@playwright/test'
import type { APIRequestContext, Page, TestInfo } from '@playwright/test'
import type { Editor, TLShapeId } from 'tldraw'
import type {
	CanvasAction,
	canvasContext,
	executeCanvasAction,
	VoiceHistory,
} from '../client/local-voice/voiceActions'

declare const editor: Editor
declare const voice: { context: typeof canvasContext; execute: typeof executeCanvasAction }

test.beforeEach(async ({ page }) => {
	await page.goto('/e2e/fixture.html')
	await page.waitForFunction(() => !!window.editor)
})

async function drag(page: Page, id: TLShapeId, dx: number, dy: number) {
	const point = await page.evaluate(
		(id) => editor.pageToScreen(editor.getShapePageBounds(id)!.center),
		id
	)
	await page.mouse.move(point.x, point.y)
	await page.mouse.down()
	await page.mouse.move(point.x + dx, point.y + dy, { steps: 8 })
	await page.mouse.up()
}

test('creates, selects, moves, styles, groups, deletes and restores native shapes', async ({
	page,
}) => {
	const result = await page.evaluate(() => {
		const act = (name: string, args: unknown) =>
			voice.execute(editor, { name, arguments: args }, voice.context(editor))
		const a = act('create_shape', { kind: 'rectangle', text: 'User', x: 0, y: 0 }).createdId!
		const b = act('create_shape', { kind: 'ellipse', text: 'Class', x: 400, y: 0 }).createdId!
		act('select_shapes', { shape_ids: [a, b] })
		act('move_shapes', { shape_ids: [a, b], dx: 50, dy: 80 })
		const moved = [editor.getShape(a)!.x, editor.getShape(b)!.x]
		act('style_shapes', { shape_ids: [a, b], color: 'blue', fill: 'solid', opacity: 0.5 })
		const styled = voice
			.context(editor)
			.shapes.map(({ color, fill, opacity }) => ({ color, fill, opacity }))
		const group = act('arrange_shapes', { shape_ids: [a, b], operation: 'group' }).createdId!
		act('move_shapes', { shape_ids: [group, a], dx: 100, dy: 0 })
		const groupedX = editor.getShapePageTransform(a).applyToPoint({ x: 0, y: 0 }).x
		act('arrange_shapes', { shape_ids: [group], operation: 'ungroup' })
		act('delete_shapes', { shape_ids: [b] })
		const deleted = !editor.getShape(b)
		act('canvas_command', { operation: 'undo' })
		const restored = !!editor.getShape(b)
		act('select_shapes', { shape_ids: [a] })
		act('canvas_command', { operation: 'zoom_in' })
		const redoKept = editor.getCanRedo()
		act('canvas_command', { operation: 'redo' })
		return {
			moved,
			styled,
			groupedX,
			deleted,
			restored,
			redoKept,
			deletedAgain: !editor.getShape(b),
		}
	})
	expect(result).toEqual({
		moved: [50, 450],
		styled: [
			{ color: 'blue', fill: 'solid', opacity: 0.5 },
			{ color: 'blue', fill: 'solid', opacity: 0.5 },
		],
		groupedX: 150,
		deleted: true,
		restored: true,
		redoKept: true,
		deletedAgain: true,
	})
})

test('supports every drawing kind, text, resizing, methods and bound arrows', async ({ page }) => {
	const result = await page.evaluate(() => {
		const act = (name: string, args: unknown) =>
			voice.execute(editor, { name, arguments: args }, voice.context(editor))
		for (const kind of [
			'rectangle',
			'ellipse',
			'diamond',
			'triangle',
			'text',
			'note',
			'frame',
			'arrow',
		]) {
			const id = act('create_shape', { kind, text: 'Start' }).createdId!
			act('set_text', { shape_id: id, text: 'Edited' })
		}
		const user = act('create_schema_box', {
			name: 'User',
			fields: ['name'],
			methods: [],
		}).createdId!
		const classId = act('create_schema_box', { name: 'Class', fields: [], methods: [] }).createdId!
		act('add_method', { schema_id: user, method_name: 'getName' })
		act('remove_method', { schema_id: user, method_name: 'getName' })
		act('resize_shape', { shape_id: user, width: 420, height: 300 })
		act('connect_schemas', { source_id: user, target_id: classId, label: 'belongs to' })
		const arrow = editor.getCurrentPageShapes().find((s) => s.meta.voiceLabel === 'belongs to')!
		const bindings = editor.getBindingsFromShape(arrow.id, 'arrow').length
		const schema = editor.getShape(user)!
		act('delete_shapes', { shape_ids: [classId] })
		const dangling = editor
			.getBindingsFromShape(arrow.id, 'arrow')
			.some((binding) => binding.toId === classId)
		return {
			drawingTexts: voice
				.context(editor)
				.shapes.filter((s) => s.kind !== 'arrow' || s.text === 'Edited')
				.map((s) => s.text),
			schema: schema.props,
			bindings,
			dangling,
		}
	})
	expect(result.drawingTexts).toHaveLength(8)
	expect(result.drawingTexts.every((text) => text === 'Edited')).toBe(true)
	expect(result.schema).toMatchObject({
		w: 420,
		h: 300,
		fields: ['name'],
		methods: [],
		color: 'black',
		fill: 'none',
	})
	expect(result.bindings).toBe(2)
	expect(result.dangling).toBe(false)
})

test('arranges shapes and rejects stale or partially missing targets atomically', async ({
	page,
}) => {
	const result = await page.evaluate(() => {
		const act = (name: string, args: unknown) =>
			voice.execute(editor, { name, arguments: args }, voice.context(editor))
		const ids = [0, 300, 600].map(
			(x) => act('create_shape', { kind: 'rectangle', x, y: x / 2 }).createdId!
		)
		act('arrange_shapes', { shape_ids: ids, operation: 'align_top' })
		const aligned = ids.map((id) => editor.getShape(id)!.y)
		for (const operation of [
			'front',
			'back',
			'forward',
			'backward',
			'distribute_horizontal',
			'distribute_vertical',
			'flip_horizontal',
			'flip_vertical',
			'stack_horizontal',
			'stack_vertical',
			'pack',
		])
			act('arrange_shapes', { shape_ids: ids, operation })
		const count = editor.getCurrentPageShapes().length
		act('arrange_shapes', { shape_ids: [ids[0]], operation: 'duplicate' })
		const duplicateCount = editor.getCurrentPageShapes().length
		const snapshot = voice.context(editor)
		act('move_shapes', { shape_ids: [ids[0]], dx: 20, dy: 0 })
		let stale = false
		try {
			voice.execute(editor, { name: 'delete_shapes', arguments: { shape_ids: ids } }, snapshot)
		} catch {
			stale = true
		}
		const before = JSON.stringify(voice.context(editor))
		let missing = false
		try {
			act('move_shapes', { shape_ids: [ids[0], 'shape:missing'], dx: 10, dy: 0 })
		} catch {
			missing = true
		}
		const unchanged = before === JSON.stringify(voice.context(editor))
		act('pan_canvas', { dx: 100, dy: 50 })
		act('canvas_command', { operation: 'select_all' })
		const selected = editor.getSelectedShapeIds().length
		act('canvas_command', { operation: 'clear_selection' })
		act('canvas_command', { operation: 'zoom_to_fit' })
		act('canvas_command', { operation: 'zoom_out' })
		act('canvas_command', { operation: 'reset_zoom' })
		return {
			aligned,
			count,
			duplicateCount,
			stale,
			missing,
			unchanged,
			selected,
			cleared: editor.getSelectedShapeIds().length === 0,
		}
	})
	expect(result).toEqual({
		aligned: [0, 0, 0],
		count: 3,
		duplicateCount: 4,
		stale: true,
		missing: true,
		unchanged: true,
		selected: 4,
		cleared: true,
	})
	await expect(page.locator('.tl-canvas').first()).toBeVisible()
})

test('captures a manual drag before the next voice edit', async ({ page }) => {
	const created = await page.evaluate(() => {
		const outcome = voice.execute(
			editor,
			{ name: 'create_shape', arguments: { kind: 'rectangle', text: 'Customer', x: 100, y: 100 } },
			voice.context(editor)
		)
		return { id: outcome.createdId!, snapshot: voice.context(editor) }
	})
	await drag(page, created.id, 35, 25)
	const moved = await page.evaluate(() => voice.context(editor))
	const result = await page.evaluate(
		({ created, moved }) => {
			const action = {
				name: 'move_shapes',
				arguments: { shape_ids: [created.id], dx: -10, dy: -15 },
			}
			let rejected = false
			try {
				voice.execute(editor, action, created.snapshot)
			} catch {
				rejected = true
			}
			voice.execute(editor, action, moved)
			return { rejected, current: voice.context(editor).shapes[0] }
		},
		{ created, moved }
	)
	expect(moved.shapes[0]).toMatchObject({ x: 135, y: 125 })
	expect(result.rejected).toBe(true)
	expect(result.current).toMatchObject({ x: 125, y: 110 })
})

test('matches quality-data assumptions for native notes, frames and grouped targets', async ({
	page,
}) => {
	const result = await page.evaluate(() => {
		const act = (name: string, args: unknown) =>
			voice.execute(editor, { name, arguments: args }, voice.context(editor))
		const note = act('create_shape', {
			kind: 'note',
			text: 'Reminder',
			x: 0,
			y: 0,
			width: 120,
			height: 80,
		}).createdId!
		const frame = act('create_shape', {
			kind: 'frame',
			text: 'Section',
			x: 400,
			y: 0,
			width: 300,
			height: 180,
		}).createdId!
		const a = act('create_shape', {
			kind: 'rectangle',
			text: 'Packing card',
			x: 900,
			y: 0,
		}).createdId!
		const b = act('create_shape', {
			kind: 'rectangle',
			text: 'Route tile',
			x: 1200,
			y: 0,
		}).createdId!
		const group = act('arrange_shapes', { shape_ids: [a, b], operation: 'group' }).createdId!
		const before = voice.context(editor)
		act('select_shapes', { shape_ids: [b] })
		act('move_shapes', { shape_ids: [a], dx: 25, dy: 0 })
		const after = voice.context(editor)
		return {
			note: after.shapes.find((s) => s.id === note),
			frame: after.shapes.find((s) => s.id === frame),
			childParent: after.shapes.find((s) => s.id === a)!.parent_id === group,
			childDistance:
				after.shapes.find((s) => s.id === a)!.x - before.shapes.find((s) => s.id === a)!.x,
			otherDistance:
				after.shapes.find((s) => s.id === b)!.x - before.shapes.find((s) => s.id === b)!.x,
			selectedTarget: after.selected_ids.length === 1 && after.selected_ids[0] === a,
		}
	})
	expect(result.note).toMatchObject({ kind: 'note', w: 200, h: 200 })
	expect(result.frame).toMatchObject({ kind: 'frame', color: 'black', fill: 'none' })
	expect(result).toMatchObject({
		childParent: true,
		childDistance: 25,
		otherDistance: 0,
		selectedTarget: true,
	})
})

test.describe('model integration', () => {
	test.skip(!process.env.VOICE_MODEL_URL, 'Requires a running local action model.')
	test.setTimeout(180_000)

	function session(page: Page, request: APIRequestContext, info: TestInfo) {
		let history: VoiceHistory = { turns: [], last_created_id: null, last_edited_id: null }
		const rows: unknown[] = []
		return async (
			input: { command: string } | { audio: string },
			expected: CanvasAction['name']
		) => {
			const canvas = await page.evaluate(() => voice.context(editor))
			const response = await request.post(`${process.env.VOICE_MODEL_URL}/command`, {
				data: { ...input, canvas, history },
				timeout: 60_000,
			})
			const result = await response.json()
			rows.push({
				input: 'command' in input ? input.command : 'synthetic speech',
				expected,
				result,
			})
			await info.attach(`model-command-${rows.length}`, {
				body: JSON.stringify(rows.at(-1), null, 2),
				contentType: 'application/json',
			})
			expect(response.ok(), JSON.stringify(result)).toBe(true)
			expect(result.action.name, JSON.stringify(result)).toBe(expected)
			const outcome = await page.evaluate(
				({ action, canvas }) => voice.execute(editor, action, canvas),
				{ action: result.action, canvas }
			)
			history = {
				turns: [
					...history.turns,
					{
						command: result.command,
						action: outcome.action,
						...(outcome.createdId ? { created_id: outcome.createdId } : {}),
					},
				].slice(-3),
				last_created_id: outcome.createdId ?? history.last_created_id,
				last_edited_id: outcome.editedId ?? history.last_edited_id,
			}
			return { result, outcome, canvas: await page.evaluate(() => voice.context(editor)) }
		}
	}

	test('executes model commands across native editing history', async ({ page, request }, info) => {
		const send = session(page, request, info)
		const create = await send(
			{ command: 'Draw a rectangle labelled Customer at x 100 and y 100.' },
			'create_shape'
		)
		expect(create.canvas.shapes).toHaveLength(1)
		expect(create.canvas.shapes[0]).toMatchObject({
			kind: 'rectangle',
			text: 'Customer',
			x: 100,
			y: 100,
		})
		const moved = await send({ command: 'Move it right by 80.' }, 'move_shapes')
		expect(moved.canvas.shapes[0]).toMatchObject({ x: 180, y: 100 })
		const styled = await send({ command: 'Make it blue.' }, 'style_shapes')
		expect(styled.canvas.shapes[0].color).toBe('blue')
		await drag(page, create.outcome.createdId!, 35, 25)
		const copy = await send({ command: 'Duplicate it.' }, 'arrange_shapes')
		expect(copy.canvas.shapes).toHaveLength(2)
		const selected = await send({ command: 'Select all.' }, 'canvas_command')
		expect(selected.canvas.selected_ids).toHaveLength(2)
		const down = await send({ command: 'Move the selected shapes down by 40.' }, 'move_shapes')
		expect(down.canvas.shapes.map((s) => s.y).sort((a, b) => a - b)).toEqual([165, 189])
		const group = await send({ command: 'Group the selected shapes.' }, 'arrange_shapes')
		expect(group.canvas.shapes.filter((s) => s.kind === 'group')).toHaveLength(1)
		const ungroup = await send({ command: 'Ungroup it.' }, 'arrange_shapes')
		expect(ungroup.canvas.shapes).toHaveLength(2)
		expect(ungroup.canvas.selected_ids).toHaveLength(2)
		const deleted = await send({ command: 'Delete the selected shapes.' }, 'delete_shapes')
		expect(deleted.canvas.shapes).toHaveLength(0)
		const undone = await send({ command: 'Undo.' }, 'canvas_command')
		expect(undone.canvas.shapes).toHaveLength(2)
		const redone = await send({ command: 'Redo.' }, 'canvas_command')
		expect(redone.canvas.shapes).toHaveLength(0)
		const noEdit = await send({ command: 'Explain what is on the canvas.' }, 'no_action')
		expect(noEdit.canvas).toEqual(redone.canvas)
	})

	test('transcribes synthetic speech and applies the model edit', async ({
		page,
		request,
	}, info) => {
		test.skip(!process.env.VOICE_SMOKE_AUDIO, 'Requires the synthetic speech fixture.')
		await page.evaluate(() => {
			voice.execute(
				editor,
				{ name: 'create_schema_box', arguments: { name: 'User', fields: [], methods: [] } },
				voice.context(editor)
			)
		})
		const audio = readFileSync(process.env.VOICE_SMOKE_AUDIO!).toString('base64')
		const edited = await session(page, request, info)({ audio }, 'add_property')
		expect(edited.result.command.toLowerCase()).toContain('email')
		expect(edited.canvas.schemas).toHaveLength(1)
		expect(edited.canvas.schemas[0]).toMatchObject({
			name: 'User',
			properties: ['email'],
			methods: [],
		})
	})
})
