import { BaseBoxShapeUtil, HTMLContainer, T, TLShape } from 'tldraw'

declare module 'tldraw' {
	export interface TLGlobalShapePropsMap {
		'voice-schema': { w: number; h: number; name: string; fields: string[]; methods: string[] }
	}
}

export type SchemaBoxShape = TLShape<'voice-schema'>

export function schemaHeight(fields: string[], methods: string[]) {
	return 52 + Math.max(1, fields.length) * 24 + Math.max(1, methods.length) * 24 + 52
}

export class SchemaBoxShapeUtil extends BaseBoxShapeUtil<SchemaBoxShape> {
	static override type = 'voice-schema' as const
	static override props = {
		w: T.positiveNumber,
		h: T.positiveNumber,
		name: T.string,
		fields: T.arrayOf(T.string),
		methods: T.arrayOf(T.string),
	}

	override getDefaultProps() {
		return { w: 280, h: schemaHeight([], []), name: 'Schema', fields: [], methods: [] }
	}

	override canEdit() {
		return false
	}

	override canResize() {
		return false
	}

	override component(shape: SchemaBoxShape) {
		return (
			<HTMLContainer className="voice-schema">
				<div className="voice-schema-name">{shape.props.name}</div>
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
