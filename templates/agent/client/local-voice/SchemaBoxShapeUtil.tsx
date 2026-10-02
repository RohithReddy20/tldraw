import {
	BaseBoxShapeUtil,
	DefaultColorStyle,
	DefaultFillStyle,
	HTMLContainer,
	T,
	TLDefaultColorStyle,
	TLDefaultFillStyle,
	TLShape,
	createShapePropsMigrationIds,
	createShapePropsMigrationSequence,
} from 'tldraw'

declare module 'tldraw' {
	export interface TLGlobalShapePropsMap {
		'voice-schema': {
			w: number
			h: number
			name: string
			fields: string[]
			methods: string[]
			color: TLDefaultColorStyle
			fill: TLDefaultFillStyle
		}
	}
}

export type SchemaBoxShape = TLShape<'voice-schema'>
const versions = createShapePropsMigrationIds('voice-schema', { AddStyles: 1 })
const migrations = createShapePropsMigrationSequence({
	sequence: [
		{
			id: versions.AddStyles,
			up(props) {
				props.color = 'black'
				props.fill = 'none'
			},
			down(props) {
				delete props.color
				delete props.fill
			},
		},
	],
})

export function schemaHeight(fields: string[], methods: string[]) {
	return 52 + Math.max(1, fields.length) * 24 + Math.max(1, methods.length) * 24 + 52
}

export class SchemaBoxShapeUtil extends BaseBoxShapeUtil<SchemaBoxShape> {
	static override type = 'voice-schema' as const
	static override migrations = migrations
	static override props = {
		w: T.positiveNumber,
		h: T.positiveNumber,
		name: T.string,
		fields: T.arrayOf(T.string),
		methods: T.arrayOf(T.string),
		color: DefaultColorStyle,
		fill: DefaultFillStyle,
	}

	override getDefaultProps() {
		return {
			w: 280,
			h: schemaHeight([], []),
			name: 'Schema',
			fields: [],
			methods: [],
			color: 'black' as const,
			fill: 'none' as const,
		}
	}

	override canEdit() {
		return false
	}

	override component(shape: SchemaBoxShape) {
		const color =
			this.editor.getCurrentTheme().colors[this.editor.getColorMode()][shape.props.color]
		const solid = shape.props.fill === 'solid'
		return (
			<HTMLContainer
				className="voice-schema"
				style={{
					borderColor: color.solid,
					backgroundColor: solid
						? color.solid
						: shape.props.fill === 'none'
							? '#faf9f5'
							: color.semi,
					color: solid ? '#fff' : color.solid,
					backgroundImage:
						shape.props.fill === 'pattern'
							? `repeating-linear-gradient(45deg, transparent 0 8px, ${color.semi} 8px 10px)`
							: undefined,
				}}
			>
				<div
					className="voice-schema-name"
					style={{
						backgroundColor: shape.props.fill === 'none' ? undefined : 'transparent',
						borderColor: 'currentColor',
					}}
				>
					{shape.props.name}
				</div>
				<div className="voice-schema-section">
					<span className="voice-schema-label">Properties</span>
					{shape.props.fields.length ? (
						shape.props.fields.map((field, i) => <div key={i}>+ {field}</div>)
					) : (
						<div className="voice-schema-empty">—</div>
					)}
				</div>
				<div className="voice-schema-section">
					<span className="voice-schema-label">Methods</span>
					{shape.props.methods.length ? (
						shape.props.methods.map((method, i) => (
							<div key={i}>+ {method.endsWith(')') ? method : `${method}()`}</div>
						))
					) : (
						<div className="voice-schema-empty">—</div>
					)}
				</div>
			</HTMLContainer>
		)
	}

	override getIndicatorPath(shape: SchemaBoxShape) {
		const path = new Path2D()
		path.rect(0, 0, shape.props.w, shape.props.h)
		return path
	}
}
