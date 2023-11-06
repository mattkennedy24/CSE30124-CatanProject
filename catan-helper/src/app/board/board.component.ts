import { Component } from '@angular/core';

@Component({
  selector: 'app-board',
  templateUrl: './board.component.html',
  styleUrls: ['./board.component.css']
})
export class BoardComponent {
  catanTiles: Array<{ numberToken: string, resource: string }> = new Array(19).fill({ numberToken: '', resource: '' });
}
