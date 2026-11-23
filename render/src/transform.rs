use crate::matrix::Matrix;
use crate::perspective_projection::PerspectiveProjection;
use swf::ColorTransform;

/// Represents the transform for a DisplayObject.
/// This includes both the transformation matrix and the color transform.
#[derive(Clone, Debug, Default)]
pub struct Transform {
    pub matrix: Matrix,
    pub tz: f64,
    pub color_transform: ColorTransform,
    pub perspective_projection: Option<PerspectiveProjection>,
}

pub struct TransformStack(Vec<Transform>);

impl TransformStack {
    pub fn new() -> Self {
        let mut stack = Vec::with_capacity(16);
        stack.push(Transform::default());
        Self(stack)
    }

    #[inline]
    pub fn push(&mut self, transform: &Transform) {
        let cur_transform = self.0.last().expect("Transform stack underflow");
        let matrix = if transform.matrix == Matrix::IDENTITY {
            cur_transform.matrix
        } else {
            cur_transform.matrix * transform.matrix
        };
        let color_transform = if transform.color_transform == ColorTransform::IDENTITY {
            cur_transform.color_transform
        } else {
            cur_transform.color_transform * transform.color_transform
        };
        let tz = cur_transform.tz + transform.tz;
        self.0.push(Transform {
            matrix,
            color_transform,
            // TODO: Merge perspective_projections from cur_transform and transform properly
            perspective_projection: Default::default(),
            tz,
        });
    }

    #[inline]
    pub fn pop(&mut self) {
        assert!(self.0.len() > 1, "Transform stack underflow");
        self.0.pop();
    }

    #[inline]
    pub fn transform(&self) -> Transform {
        self.0[self.0.len() - 1].clone()
    }

    #[inline]
    pub fn transform_ref(&self) -> &Transform {
        &self.0[self.0.len() - 1]
    }
}

impl Default for TransformStack {
    fn default() -> Self {
        TransformStack::new()
    }
}
