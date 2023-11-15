import { Component } from '@angular/core';
import { Board } from '../models/board.model';
import { Tile } from '../models/tile.model';
import { Router } from '@angular/router';


@Component({
  selector: 'app-board',
  templateUrl: './board.component.html',
  styleUrls: ['./board.component.css']
})
export class BoardComponent {
  constructor(private router: Router) {}

  tiles: Tile[] = Array.from({ length: 19 }, (_, i) => new Tile(0, '', i));
  // Add a method to handle form submission
  submitForm() {
    console.log('Form submitted');
    
    const catanBoard = new Board(this.tiles)
    this.router.navigate(['/chat', { catanBoard }]);
  }
}
