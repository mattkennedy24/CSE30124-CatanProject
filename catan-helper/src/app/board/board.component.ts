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

  tiles: Tile[] = Array.from({ length: 19 }, (_, i) => new Tile('', '', i, ''));
  // Add a method to handle form submission
  submitForm() {
    console.log('Form submitted');
    
    const catanBoard = new Board(this.tiles)
    // console.log(catanBoard)
    // this.router.navigate(['/chat', { catanBoard }]);
    this.router.navigate(['/chat'], { queryParams: { catanBoard: JSON.stringify(catanBoard) } });

  }
}
