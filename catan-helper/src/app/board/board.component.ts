import { Component } from '@angular/core';
import { Board } from '../models/board.model';
import { Tile } from '../models/tile.model';

@Component({
  selector: 'app-board',
  templateUrl: './board.component.html',
  styleUrls: ['./board.component.css']
})
export class BoardComponent {
  tiles: Tile[] = Array.from({ length: 19 }, (_, i) => new Tile(0, '', i));
  catanBoard = new Board(this.tiles)
  // Add a method to handle form submission
  submitForm() {
    console.log('Form submitted');
    
    this.catanBoard = new Board(this.tiles)
    console.log(this.catanBoard);
  }
}
