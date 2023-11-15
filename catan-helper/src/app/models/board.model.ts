import { Tile } from './tile.model';

export class Board {
  tiles: Tile[];

  constructor(tiles: Tile[]) {
    this.tiles = tiles;
  }
}