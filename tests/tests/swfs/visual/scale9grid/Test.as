package {
    import flash.display.*;
    import flash.geom.*;

    // 9-slice scaling (scale9Grid): which content is sliced, and how strokes and fills are
    // drawn when it is.
    public class Test extends Sprite {
        private static const GRID:Rectangle = new Rectangle(14, 14, 32, 12);

        // A 60x40 box with rounded corners, an outline and a dot in its left column.
        private static function box(g:Graphics):void {
            g.lineStyle(2, 0x000000);
            g.beginFill(0x66AAFF);
            g.drawRoundRect(0, 0, 60, 40, 24, 24);
            g.endFill();
            g.lineStyle();
            g.beginFill(0xFF3300);
            g.drawRect(4, 16, 8, 8);
            g.endFill();
        }

        // Gradient pieces: a corner cell, the center cell, a row across all columns and a
        // column across two rows.
        private static function gradients(g:Graphics):void {
            piece(g, 0, 0, 14, 14);
            piece(g, 14, 14, 32, 12);
            piece(g, 0, 26, 60, 14);
            piece(g, 46, 0, 14, 26);
        }

        private static function piece(g:Graphics, x:Number, y:Number, w:Number, h:Number):void {
            var m:Matrix = new Matrix();
            m.createGradientBox(w, h, 0, x, y);
            g.beginGradientFill(GradientType.LINEAR, [0xFF0000, 0x0000FF], [1, 1], [0, 255], m);
            g.drawRect(x, y, w, h);
            g.endFill();
        }

        // The same pieces with a bitmap fill.
        private static function bitmaps(g:Graphics):void {
            var b:BitmapData = new BitmapData(4, 4, false, 0xFFFFFF);
            for (var y:int = 0; y < 4; y++)
                for (var x:int = 0; x < 4; x++)
                    if ((x + y) % 2 == 0) b.setPixel(x, y, 0x00AA00);
            bitmapPiece(g, b, 0, 0, 14, 14);
            bitmapPiece(g, b, 14, 14, 32, 12);
            bitmapPiece(g, b, 0, 26, 60, 14);
            bitmapPiece(g, b, 46, 0, 14, 26);
        }

        private static function bitmapPiece(g:Graphics, b:BitmapData, x:Number, y:Number, w:Number, h:Number):void {
            var m:Matrix = new Matrix();
            m.scale(2, 2);
            m.translate(x, y);
            g.beginBitmapFill(b, m, true, false);
            g.drawRect(x, y, w, h);
            g.endFill();
        }

        private function place(s:DisplayObject, x:Number, y:Number, sx:Number, sy:Number, rotation:Number = 0):DisplayObject {
            s.scale9Grid = GRID;
            s.scaleX = sx;
            s.scaleY = sy;
            s.rotation = rotation;
            s.x = x;
            s.y = y;
            addChild(s);
            return s;
        }

        public function Test() {
            // The sprite's own drawing: sliced.
            var a:Sprite = new Sprite();
            box(a.graphics);
            place(a, 5, 5, 2, 1.5);

            // A shape in the sprite: sliced.
            var b:Sprite = new Sprite();
            var shape:Shape = new Shape();
            box(shape.graphics);
            b.addChild(shape);
            place(b, 135, 5, 2, 1.5);

            // A sprite in the sprite: scaled normally.
            var c:Sprite = new Sprite();
            var inner:Sprite = new Sprite();
            box(inner.graphics);
            c.addChild(inner);
            place(c, 280, 5, 1.5, 1.5);

            // A shape with its own grid: sliced.
            var d:Shape = new Shape();
            box(d.graphics);
            place(d, 5, 75, 2, 1.5);

            // Rotated: scaled normally.
            var e:Sprite = new Sprite();
            box(e.graphics);
            place(e, 160, 70, 1.5, 1.5, 15);

            // Unscaled, in a scaled parent: scaled normally (only the sprite's own scale counts).
            var parent:Sprite = new Sprite();
            var f:Sprite = new Sprite();
            box(f.graphics);
            f.scale9Grid = GRID;
            parent.addChild(f);
            parent.scaleX = parent.scaleY = 1.5;
            parent.x = 280;
            parent.y = 75;
            addChild(parent);

            // Gradient and bitmap fills follow their piece of the grid.
            var g:Sprite = new Sprite();
            gradients(g.graphics);
            place(g, 5, 150, 2.5, 2.5);

            var h:Sprite = new Sprite();
            bitmaps(h.graphics);
            place(h, 165, 150, 2.5, 2.5);
        }
    }
}
