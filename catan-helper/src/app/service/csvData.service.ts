import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, map } from 'rxjs';
import { Papa } from 'ngx-papaparse'; // Install ngx-papaparse via npm

@Injectable({
  providedIn: 'root',
})
export class CsvService {
  constructor(private http: HttpClient, private papa: Papa) {}

  getCsvData(csvFileName: string): Observable<any> {
    const csvFileUrl = `../../assets/data/${csvFileName}`;
    return this.http.get(csvFileUrl, { responseType: 'text' }).pipe(
      map((csvData: string) => {
        const parsedData = this.papa.parse(csvData, { header: true });
        return parsedData.data;
      })
    );
  }
}
